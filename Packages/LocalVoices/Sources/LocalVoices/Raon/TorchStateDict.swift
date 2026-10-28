import Foundation
import MLX

/// Reads a PyTorch `torch.save` checkpoint that holds a flat state_dict of tensors, without Python.
///
/// Such a file is a zip archive with `<name>/data.pkl` (a pickle describing the dictionary) and one raw
/// little-endian file per tensor storage under `<name>/data/`. Only the small part of pickle that
/// `torch.save` writes for a state_dict is understood; anything else is an error rather than a guess.
/// Used for speechbrain's HiFi-GAN `generator.ckpt`, which has no safetensors copy.
enum TorchStateDict {
    static func load(_ url: URL) throws -> [String: MLXArray] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let archive = try StoredZip(data)
        guard let pickleName = archive.names.first(where: { $0.hasSuffix("/data.pkl") || $0 == "data.pkl" }) else {
            throw LocalVoiceError("\(url.lastPathComponent) is not a PyTorch checkpoint.")
        }
        let root = String(pickleName.dropLast("data.pkl".count))
        var unpickler = Unpickler(try archive.contents(pickleName))
        guard case .dict(let entries) = try unpickler.run() else {
            throw LocalVoiceError("\(url.lastPathComponent) does not hold a state_dict.")
        }

        var result: [String: MLXArray] = [:]
        for (key, value) in entries {
            guard case .string(let name) = key, case .tensor(let tensor) = value else { continue }
            let storage = try archive.contents(root + "data/" + tensor.storageKey)
            let count = tensor.shape.reduce(1, *)
            let start = tensor.offset * tensor.dtype.size
            let byteCount = count * tensor.dtype.size
            guard start + byteCount <= storage.count else {
                throw LocalVoiceError("Tensor \(name) runs past its storage.")
            }
            var expected = 1
            for (size, stride) in zip(tensor.shape, tensor.stride).reversed() {
                guard size == 1 || stride == expected else {
                    throw LocalVoiceError("Tensor \(name) is not contiguous.")
                }
                expected *= size
            }
            let bytes = storage.subdata(in: (storage.startIndex + start) ..< (storage.startIndex + start + byteCount))
            result[name] = MLXArray(bytes, tensor.shape, dtype: tensor.dtype)
        }
        return result
    }

    fileprivate struct TensorRef {
        let storageKey: String
        let dtype: DType
        let offset: Int
        let shape: [Int]
        let stride: [Int]
    }

    fileprivate indirect enum Value {
        case none
        case bool(Bool)
        case int(Int)
        case float(Double)
        case string(String)
        case tuple([Value])
        case list([Value])
        case dict([(Value, Value)])
        case global(String, String)
        case storage(key: String, dtype: DType)
        case tensor(TensorRef)
        case mark
    }

    /// The pickle opcodes `torch.save` uses for a dictionary of tensors (protocol 2, plus a few
    /// protocol 4 ones that newer writers emit).
    fileprivate struct Unpickler {
        private let bytes: [UInt8]
        private var position = 0
        private var stack: [Value] = []
        private var memo: [Int: Value] = [:]

        init(_ data: Data) {
            bytes = [UInt8](data)
        }

        mutating func run() throws -> Value {
            while position < bytes.count {
                let op = bytes[position]
                position += 1
                switch op {
                case 0x80: position += 1 // PROTO
                case 0x95: position += 8 // FRAME
                case 0x2E: // STOP
                    guard let top = stack.popLast() else { throw failure("empty stack at STOP") }
                    return top
                case 0x7D: stack.append(.dict([])) // EMPTY_DICT
                case 0x5D: stack.append(.list([])) // EMPTY_LIST
                case 0x29: stack.append(.tuple([])) // EMPTY_TUPLE
                case 0x28: stack.append(.mark) // MARK
                case 0x4E: stack.append(.none) // NONE
                case 0x88: stack.append(.bool(true)) // NEWTRUE
                case 0x89: stack.append(.bool(false)) // NEWFALSE
                case 0x4A: stack.append(.int(Int(Int32(bitPattern: UInt32(try readUInt(4)))))) // BININT
                case 0x4B: stack.append(.int(try readUInt(1))) // BININT1
                case 0x4D: stack.append(.int(try readUInt(2))) // BININT2
                case 0x8A: // LONG1
                    let length = try readUInt(1)
                    let raw = try read(length)
                    var value = 0
                    for (index, byte) in raw.enumerated() { value |= Int(byte) << (8 * index) }
                    if length > 0, length < 8, raw[length - 1] & 0x80 != 0 { value -= 1 << (8 * length) }
                    stack.append(.int(value))
                case 0x47: // BINFLOAT, big-endian
                    let raw = try read(8)
                    let bits = raw.reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
                    stack.append(.float(Double(bitPattern: bits)))
                case 0x58: stack.append(.string(try readString(try readUInt(4)))) // BINUNICODE
                case 0x8C: stack.append(.string(try readString(try readUInt(1)))) // SHORT_BINUNICODE
                case 0x55: stack.append(.string(try readString(try readUInt(1)))) // SHORT_BINSTRING
                case 0x54: stack.append(.string(try readString(try readUInt(4)))) // BINSTRING
                case 0x71, 0x72, 0x94: // BINPUT, LONG_BINPUT, MEMOIZE
                    var index = memo.count
                    if op == 0x71 { index = try readUInt(1) } else if op == 0x72 { index = try readUInt(4) }
                    let value = try top()
                    memo[index] = value
                case 0x68: stack.append(try recall(try readUInt(1))) // BINGET
                case 0x6A: stack.append(try recall(try readUInt(4))) // LONG_BINGET
                case 0x63: // GLOBAL
                    let module = try readLine()
                    let name = try readLine()
                    stack.append(.global(module, name))
                case 0x93: // STACK_GLOBAL
                    guard case .string(let name) = try pop(), case .string(let module) = try pop() else {
                        throw failure("bad STACK_GLOBAL")
                    }
                    stack.append(.global(module, name))
                case 0x85: stack.append(.tuple([try pop()])) // TUPLE1
                case 0x86: // TUPLE2
                    let second = try pop()
                    stack.append(.tuple([try pop(), second]))
                case 0x87: // TUPLE3
                    let third = try pop()
                    let second = try pop()
                    stack.append(.tuple([try pop(), second, third]))
                case 0x74: stack.append(.tuple(try popToMark())) // TUPLE
                case 0x61: // APPEND
                    let item = try pop()
                    guard case .list(let items) = try pop() else { throw failure("APPEND to a non-list") }
                    stack.append(.list(items + [item]))
                case 0x65: // APPENDS
                    let items = try popToMark()
                    guard case .list(let existing) = try pop() else { throw failure("APPENDS to a non-list") }
                    stack.append(.list(existing + items))
                case 0x73: // SETITEM
                    let value = try pop()
                    let key = try pop()
                    guard case .dict(let entries) = try pop() else { throw failure("SETITEM on a non-dict") }
                    stack.append(.dict(entries + [(key, value)]))
                case 0x75: // SETITEMS
                    let items = try popToMark()
                    guard case .dict(var entries) = try pop(), items.count % 2 == 0 else {
                        throw failure("SETITEMS on a non-dict")
                    }
                    for index in stride(from: 0, to: items.count, by: 2) {
                        entries.append((items[index], items[index + 1]))
                    }
                    stack.append(.dict(entries))
                case 0x51: stack.append(try persistentLoad(try pop())) // BINPERSID
                case 0x52: // REDUCE
                    let arguments = try pop()
                    let callable = try pop()
                    stack.append(try reduce(callable, arguments))
                case 0x62: // BUILD: only sets attributes such as a state_dict's _metadata
                    _ = try pop()
                default:
                    throw failure(String(format: "unsupported opcode 0x%02X", op))
                }
            }
            throw failure("no STOP")
        }

        private func failure(_ message: String) -> LocalVoiceError {
            LocalVoiceError("Could not read the checkpoint: \(message).")
        }

        private mutating func read(_ count: Int) throws -> ArraySlice<UInt8> {
            guard position + count <= bytes.count else { throw failure("truncated") }
            defer { position += count }
            return bytes[position ..< position + count]
        }

        private mutating func readUInt(_ count: Int) throws -> Int {
            var value = 0
            for (index, byte) in try read(count).enumerated() { value |= Int(byte) << (8 * index) }
            return value
        }

        private mutating func readString(_ count: Int) throws -> String {
            String(decoding: try read(count), as: UTF8.self)
        }

        private mutating func readLine() throws -> String {
            guard let end = bytes[position...].firstIndex(of: 0x0A) else { throw failure("unterminated GLOBAL") }
            defer { position = end + 1 }
            return String(decoding: bytes[position ..< end], as: UTF8.self)
        }

        private func top() throws -> Value {
            guard let value = stack.last else { throw failure("empty stack") }
            return value
        }

        private mutating func pop() throws -> Value {
            guard let value = stack.popLast() else { throw failure("empty stack") }
            return value
        }

        private func recall(_ index: Int) throws -> Value {
            guard let value = memo[index] else { throw failure("missing memo \(index)") }
            return value
        }

        private mutating func popToMark() throws -> [Value] {
            var items: [Value] = []
            while let value = stack.popLast() {
                if case .mark = value { return items.reversed() }
                items.append(value)
            }
            throw failure("no MARK")
        }

        /// torch.save stores each storage as ('storage', <StorageType>, key, location, size).
        private func persistentLoad(_ id: Value) throws -> Value {
            guard case .tuple(let parts) = id, parts.count >= 3, case .string("storage") = parts[0],
                  case .global(_, let type) = parts[1], case .string(let key) = parts[2] else {
                throw failure("unknown persistent id")
            }
            let dtypes: [String: DType] = [
                "FloatStorage": .float32, "HalfStorage": .float16, "BFloat16Storage": .bfloat16,
                "LongStorage": .int64, "IntStorage": .int32, "BoolStorage": .bool
            ]
            guard let dtype = dtypes[type] else { throw failure("unsupported storage \(type)") }
            return .storage(key: key, dtype: dtype)
        }

        private func reduce(_ callable: Value, _ arguments: Value) throws -> Value {
            guard case .global(let module, let name) = callable, case .tuple(let args) = arguments else {
                throw failure("bad REDUCE")
            }
            switch (module, name) {
            case ("collections", "OrderedDict"):
                return .dict([])
            case ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor"):
                guard args.count >= 4, case .storage(let key, let dtype) = args[0], case .int(let offset) = args[1],
                      case .tuple(let shape) = args[2], case .tuple(let stride) = args[3] else {
                    throw failure("bad tensor")
                }
                func ints(_ values: [Value]) throws -> [Int] {
                    try values.map {
                        guard case .int(let value) = $0 else { throw failure("bad tensor shape") }
                        return value
                    }
                }
                return .tensor(TensorRef(storageKey: key, dtype: dtype, offset: offset,
                                         shape: try ints(shape), stride: try ints(stride)))
            case ("torch._utils", "_rebuild_parameter"):
                guard let data = args.first else { throw failure("bad parameter") }
                return data
            default:
                throw failure("unsupported call \(module).\(name)")
            }
        }
    }
}

/// The parts of a zip archive `torch.save` writes: stored (uncompressed) entries, optionally zip64.
private struct StoredZip {
    private let data: Data
    private var entries: [String: (offset: Int, size: Int)] = [:]

    var names: [String] { Array(entries.keys) }

    init(_ data: Data) throws {
        self.data = data
        let bad = LocalVoiceError("The checkpoint is not a readable zip archive.")
        guard data.count >= 22 else { throw bad }
        var end = data.count - 22
        while end >= 0, uint(end, 4) != 0x0605_4B50 { end -= 1 }
        guard end >= 0 else { throw bad }
        var count = uint(end + 10, 2)
        var directory = uint(end + 16, 4)
        if count == 0xFFFF || directory == 0xFFFF_FFFF {
            // zip64: the locator just before the end record points at the zip64 end record.
            guard end >= 20, uint(end - 20, 4) == 0x0706_4B50 else { throw bad }
            let record = uint(end - 12, 8)
            guard uint(record, 4) == 0x0606_4B50 else { throw bad }
            count = uint(record + 32, 8)
            directory = uint(record + 48, 8)
        }
        var cursor = directory
        for _ in 0 ..< count {
            guard uint(cursor, 4) == 0x0201_4B50 else { throw bad }
            let method = uint(cursor + 10, 2)
            var size = uint(cursor + 24, 4)
            let nameLength = uint(cursor + 28, 2)
            let extraLength = uint(cursor + 30, 2)
            let commentLength = uint(cursor + 32, 2)
            var local = uint(cursor + 42, 4)
            let name = String(decoding: data[(data.startIndex + cursor + 46) ..< (data.startIndex + cursor + 46 + nameLength)], as: UTF8.self)
            // zip64 extra field: the 64-bit values appear in order for each 32-bit field that is 0xFFFFFFFF.
            var extra = cursor + 46 + nameLength
            let extraEnd = extra + extraLength
            while extra + 4 <= extraEnd {
                let id = uint(extra, 2)
                let length = uint(extra + 2, 2)
                if id == 0x0001 {
                    var field = extra + 4
                    if size == 0xFFFF_FFFF { size = uint(field, 8); field += 8 }
                    if uint(cursor + 20, 4) == 0xFFFF_FFFF { field += 8 }
                    if local == 0xFFFF_FFFF { local = uint(field, 8) }
                }
                extra += 4 + length
            }
            guard method == 0 else { throw LocalVoiceError("The checkpoint entry \(name) is compressed.") }
            let localName = uint(local + 26, 2)
            let localExtra = uint(local + 28, 2)
            entries[name] = (local + 30 + localName + localExtra, size)
            cursor += 46 + nameLength + extraLength + commentLength
        }
    }

    func contents(_ name: String) throws -> Data {
        guard let entry = entries[name], entry.offset + entry.size <= data.count else {
            throw LocalVoiceError("The checkpoint has no \(name).")
        }
        let start = data.startIndex + entry.offset
        return data.subdata(in: start ..< start + entry.size)
    }

    private func uint(_ offset: Int, _ count: Int) -> Int {
        var value = 0
        for index in 0 ..< count {
            value |= Int(data[data.startIndex + offset + index]) << (8 * index)
        }
        return value
    }
}
