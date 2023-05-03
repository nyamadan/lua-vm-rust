function add(a, b)
    print("a => ?")
    print(a)
    print("b => ?")
    print(b)
    return a + b
end

local tbl = { x = 123, y = "hello" };
print("tbl.x => ?")
print(tbl.x)
print("tbl.y => ?")
print(tbl.y)

print("Loop Test")
for i = 0, 3 do
    print(i)
end

print("Add Function Test")
local x = 10
local y = 20
local z = add(x, y) + 1
print("(x = 10) + (y = 20) + 1 => ?")
print(z)
