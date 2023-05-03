use num_derive::*;
use num_traits::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const MAX_REGISTERS: usize = 250;
const MAX_UPVALUES: usize = 250;

#[derive(Debug, Clone)]
enum LuaValue {
    String(String),
    Float(f64),
    Nil(),
    Integer(i64),
    Closure(Function),
    Table(HashMap<String, LuaValue>),
}

#[derive(Debug, Clone)]
struct Instruction {
    op: LuaOp,
    arg: u32,
}

#[derive(Debug, Clone)]
struct Constant {
    t: LuaVariables,
    value: LuaValue,
}

#[derive(Debug, Clone)]
struct UpValue {
    instack: u8,
    idx: u8,
    kind: u8,
}

#[derive(Debug, Clone)]
enum Function {
    LuaFunction(LuaFunction),
    CFunction(CFunction),
}

#[derive(Debug, Clone)]
struct LuaMachineStatus {
    register: [Option<LuaValue>; MAX_REGISTERS],
    upvalues: [HashMap<String, LuaValue>; MAX_UPVALUES],
}

fn print(lms: LuaMachineStatus, base: usize, nargs: usize, nresults: usize) -> LuaMachineStatus {
    let register = lms.register;

    for i in 1..=nargs {
        match register[base + i].as_ref().unwrap() {
            LuaValue::String(x) => {
                println!("{}", x);
            }
            LuaValue::Integer(x) => {
                println!("{}", x);
            }
            LuaValue::Float(x) => {
                println!("{}", x);
            }
            x => {
                println!("{:?}", x);
            }
        }
    }

    LuaMachineStatus {
        register,
        upvalues: lms.upvalues,
    }
}

#[derive(Debug, Clone)]
struct CFunction {
    func: fn(LuaMachineStatus, base: usize, nargs: usize, nresults: usize) -> LuaMachineStatus,
}

#[derive(Debug, Clone)]
struct LuaFunction {
    source: String,
    line_defined: i64,
    last_line_defined: i64,
    num_params: u8,
    is_var_arg: u8,
    max_stack_size: u8,
    code: Vec<Instruction>,
    constants: Vec<Constant>,
    protos: Vec<LuaFunction>,
    upvalues: Vec<UpValue>,
    line_info: Vec<u8>,
    abs_line_info: Vec<AbsLineInfo>,
    loc_vars: Vec<LocVar>,
    upvalue_names: Vec<String>,
}

#[derive(Debug, Clone)]
struct AbsLineInfo {
    pc: i64,
    line: i64,
}
#[derive(Debug, Clone)]
struct LocVar {
    var_name: String,
    start_pc: i64,
    end_pc: i64,
}

#[derive(Debug, Clone, PartialEq, FromPrimitive)]
enum LuaVariables {
    VNIL = 0,
    VFALSE = 1,
    VTRUE = 17,
    VNUMFLT = 19,
    VNUMINT = 3,
    VSHRSTR = 4,
    VLNGSTR = 20,
}

#[derive(Debug, Clone, PartialEq, FromPrimitive)]
enum LuaOp {
    MOVE,       /*	A B	R[A] := R[B]					*/
    LOADI,      /*	A sBx	R[A] := sBx					*/
    LOADF,      /*	A sBx	R[A] := (lua_Number)sBx				*/
    LOADK,      /*	A Bx	R[A] := K[Bx]					*/
    LOADKX,     /*	A	R[A] := K[extra arg]				*/
    LOADFALSE,  /*	A	R[A] := false					*/
    LFALSESKIP, /*A	R[A] := false; pc++	(*)			*/
    LOADTRUE,   /*	A	R[A] := true					*/
    LOADNIL,    /*	A B	R[A], R[A+1], ..., R[A+B] := nil		*/
    GETUPVAL,   /*	A B	R[A] := UpValue[B]				*/
    SETUPVAL,   /*	A B	UpValue[B] := R[A]				*/

    GETTABUP, /*	A B C	R[A] := UpValue[B][K[C]:string]			*/
    GETTABLE, /*	A B C	R[A] := R[B][R[C]]				*/
    GETI,     /*	A B C	R[A] := R[B][C]					*/
    GETFIELD, /*	A B C	R[A] := R[B][K[C]:string]			*/

    SETTABUP, /*	A B C	UpValue[A][K[B]:string] := RK(C)		*/
    SETTABLE, /*	A B C	R[A][R[B]] := RK(C)				*/
    SETI,     /*	A B C	R[A][B] := RK(C)				*/
    SETFIELD, /*	A B C	R[A][K[B]:string] := RK(C)			*/

    NEWTABLE, /*	A B C k	R[A] := {}					*/

    SELF, /*	A B C	R[A+1] := R[B]; R[A] := R[B][RK(C):string]	*/

    ADDI, /*	A B sC	R[A] := R[B] + sC				*/

    ADDK,  /*	A B C	R[A] := R[B] + K[C]:number			*/
    SUBK,  /*	A B C	R[A] := R[B] - K[C]:number			*/
    MULK,  /*	A B C	R[A] := R[B] * K[C]:number			*/
    MODK,  /*	A B C	R[A] := R[B] % K[C]:number			*/
    POWK,  /*	A B C	R[A] := R[B] ^ K[C]:number			*/
    DIVK,  /*	A B C	R[A] := R[B] / K[C]:number			*/
    IDIVK, /*	A B C	R[A] := R[B] // K[C]:number			*/

    BANDK, /*	A B C	R[A] := R[B] & K[C]:integer			*/
    BORK,  /*	A B C	R[A] := R[B] | K[C]:integer			*/
    BXORK, /*	A B C	R[A] := R[B] ~ K[C]:integer			*/

    SHRI, /*	A B sC	R[A] := R[B] >> sC				*/
    SHLI, /*	A B sC	R[A] := sC << R[B]				*/

    ADD,  /*	A B C	R[A] := R[B] + R[C]				*/
    SUB,  /*	A B C	R[A] := R[B] - R[C]				*/
    MUL,  /*	A B C	R[A] := R[B] * R[C]				*/
    MOD,  /*	A B C	R[A] := R[B] % R[C]				*/
    POW,  /*	A B C	R[A] := R[B] ^ R[C]				*/
    DIV,  /*	A B C	R[A] := R[B] / R[C]				*/
    IDIV, /*	A B C	R[A] := R[B] // R[C]				*/

    BAND, /*	A B C	R[A] := R[B] & R[C]				*/
    BOR,  /*	A B C	R[A] := R[B] | R[C]				*/
    BXOR, /*	A B C	R[A] := R[B] ~ R[C]				*/
    SHL,  /*	A B C	R[A] := R[B] << R[C]				*/
    SHR,  /*	A B C	R[A] := R[B] >> R[C]				*/

    MMBIN,  /*	A B C	call C metamethod over R[A] and R[B]	(*)	*/
    MMBINI, /*	A sB C k	call C metamethod over R[A] and sB	*/
    MMBINK, /*	A B C k		call C metamethod over R[A] and K[B]	*/

    UNM,  /*	A B	R[A] := -R[B]					*/
    BNOT, /*	A B	R[A] := ~R[B]					*/
    NOT,  /*	A B	R[A] := not R[B]				*/
    LEN,  /*	A B	R[A] := #R[B] (length operator)			*/

    CONCAT, /*	A B	R[A] := R[A].. ... ..R[A + B - 1]		*/

    CLOSE, /*	A	close all upvalues >= R[A]			*/
    TBC,   /*	A	mark variable A "to be closed"			*/
    JMP,   /*	sJ	pc += sJ					*/
    EQ,    /*	A B k	if ((R[A] == R[B]) ~= k) then pc++		*/
    LT,    /*	A B k	if ((R[A] <  R[B]) ~= k) then pc++		*/
    LE,    /*	A B k	if ((R[A] <= R[B]) ~= k) then pc++		*/

    EQK, /*	A B k	if ((R[A] == K[B]) ~= k) then pc++		*/
    EQI, /*	A sB k	if ((R[A] == sB) ~= k) then pc++		*/
    LTI, /*	A sB k	if ((R[A] < sB) ~= k) then pc++			*/
    LEI, /*	A sB k	if ((R[A] <= sB) ~= k) then pc++		*/
    GTI, /*	A sB k	if ((R[A] > sB) ~= k) then pc++			*/
    GEI, /*	A sB k	if ((R[A] >= sB) ~= k) then pc++		*/

    TEST,    /*	A k	if (not R[A] == k) then pc++			*/
    TESTSET, /*	A B k	if (not R[B] == k) then pc++ else R[A] := R[B] (*) */

    CALL,     /*	A B C	R[A], ... ,R[A+C-2] := R[A](R[A+1], ... ,R[A+B-1]) */
    TAILCALL, /*	A B C k	return R[A](R[A+1], ... ,R[A+B-1])		*/

    RETURN,  /*	A B C k	return R[A], ... ,R[A+B-2]	(see note)	*/
    RETURN0, /*		return						*/
    RETURN1, /*	A	return R[A]					*/

    FORLOOP, /*	A Bx	update counters; if loop continues then pc-=Bx; */
    FORPREP, /*	A Bx	<check values and prepare counters>;
             if not to run then pc+=Bx+1;			*/

    TFORPREP, /*	A Bx	create upvalue for R[A + 3]; pc+=Bx		*/
    TFORCALL, /*	A C	R[A+4], ... ,R[A+3+C] := R[A](R[A+1], R[A+2]);	*/
    TFORLOOP, /*	A Bx	if R[A+2] ~= nil then { R[A]=R[A+2]; pc -= Bx }	*/

    SETLIST, /*	A B C k	R[A][C+i] := R[A+i], 1 <= i <= B		*/

    CLOSURE, /*	A Bx	R[A] := closure(KPROTO[Bx])			*/

    VARARG, /*	A C	R[A], R[A+1], ..., R[A+C-2] = vararg		*/

    VARARGPREP, /*A	(adjust vararg parameters)			*/

    EXTRAARG, /*	Ax	extra (larger) argument for previous opcode	*/
}

struct LuaMachine {
    instruction_size: usize,
    integer_size: usize,
    number_size: usize,
}

fn main() -> io::Result<()> {
    let path = Path::new("add.lua.out");
    let mut file = File::open(&path)?;
    let mut bytecodes = Vec::new();
    file.read_to_end(&mut bytecodes)?;

    let mut pos = 0;

    let signature = load_bytes(&bytecodes, &mut pos, 4);
    if &signature == b"\x1bLua" {
        println!("The file has a valid Lua bytecode signature.");
    } else {
        panic!("The file does not have a valid Lua bytecode signature.");
    }

    let version = load_u8(&bytecodes, &mut pos);
    if version == 0x54 {
        println!("The version is correct.");
    } else {
        panic!("The version is incorrect.");
    }

    let format = load_u8(&bytecodes, &mut pos);
    if format == 0 {
        println!("The format is correct.");
    } else {
        panic!("The format is incorrect.");
    }

    let luac_data = &[0x19, 0x93, 0x0d, 0x0a, 0x1a, 0x0a];
    if load_bytes(&bytecodes, &mut pos, luac_data.len()) == luac_data {
        println!("The Luac data section is present and has a valid format.");
    } else {
        panic!("The Luac data section is missing or has an invalid format.");
    }

    let instruction_size = load_u8(&bytecodes, &mut pos) as usize;
    println!("Instruction size: {}", instruction_size);
    let integer_size = load_u8(&bytecodes, &mut pos) as usize;
    println!("Integer size: {}", integer_size);
    let number_size = load_u8(&bytecodes, &mut pos) as usize;
    println!("Number size: {}", number_size);

    let lm = LuaMachine {
        instruction_size,
        integer_size,
        number_size,
    };

    let luac_int = load_i64(&bytecodes, &mut pos);
    if luac_int == 0x5678 {
        println!("The Luac integer format is correct.");
    } else {
        panic!("The Luac integer format is incorrect.");
    }

    let luac_num = load_f64(&bytecodes, &mut pos);
    if luac_num == 370.5 {
        println!("The Luac number format is correct.");
    } else {
        panic!("The Luac number format is incorrect.");
    }

    let size_lua_closure = load_u8(&bytecodes, &mut pos);
    println!("Size of Lua closure: {}", size_lua_closure);

    let func = load_functions(&bytecodes, &mut pos, &lm, None);

    let register: [Option<LuaValue>; MAX_REGISTERS] = vec![None; MAX_REGISTERS].try_into().unwrap();
    let mut global: HashMap<String, LuaValue> = HashMap::new();
    global.insert(
        String::from("print"),
        LuaValue::Closure(Function::CFunction(CFunction { func: print })),
    );
    let mut upvalues: [HashMap<String, LuaValue>; MAX_UPVALUES] =
        vec![HashMap::new(); MAX_UPVALUES].try_into().unwrap();
    upvalues[0] = global;

    let base = 0;
    let nargs = 0;
    let nresults = 0;
    let lms = LuaMachineStatus { register, upvalues };
    run_function(&Function::LuaFunction(func), lms, base, nargs, nresults);
    Ok(())
}

fn run_lua_function(
    func: &LuaFunction,
    lms: LuaMachineStatus,
    base: usize,
    nargs: usize,
    nresults: usize,
) -> LuaMachineStatus {
    let base = base + 1;
    let mut register = lms.register;
    let mut upvalues = lms.upvalues;

    let mut idx = 0;
    while idx < func.code.len() {
        let inst = &func.code[idx];
        match inst.op {
            LuaOp::VARARGPREP => {
                // 未実装
            }
            LuaOp::ADDI => {
                let a = (inst.arg & 0xff) as usize;
                let b = ((inst.arg >> 9) & 0xff) as usize;
                let c = ((inst.arg >> 17) & 0xff) as usize;
                let sc = c as i64 - 127;

                let rb = register[b + base].as_ref().clone().unwrap();
                if let &LuaValue::Integer(rb) = rb {
                    register[a + base] = Some(LuaValue::Integer(rb + sc));
                    idx += 1;
                };
            }
            LuaOp::ADD => {
                let a = (inst.arg & 0xff) as usize;
                let b = ((inst.arg >> 9) & 0xff) as usize;
                let c = ((inst.arg >> 17) & 0xff) as usize;
                let rb = register[b + base].as_ref().unwrap();
                let rc = register[c + base].as_ref().unwrap();
                match rb {
                    LuaValue::Integer(rb) => match rc {
                        LuaValue::Integer(rc) => {
                            register[a + base] = Some(LuaValue::Integer(rb + rc));
                            idx += 1;
                        }
                        LuaValue::Float(rc) => {
                            register[a + base] = Some(LuaValue::Float(*rb as f64 + rc));
                            idx += 1;
                        }
                        _ => {}
                    },
                    LuaValue::Float(rb) => match rc {
                        LuaValue::Integer(rc) => {
                            register[a + base] = Some(LuaValue::Float(rb + *rc as f64));
                            idx += 1;
                        }
                        LuaValue::Float(rc) => {
                            register[a + base] = Some(LuaValue::Float(rb + rc));
                            idx += 1;
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            LuaOp::CALL => {
                let a = (inst.arg & 0xff) as usize;
                let b = ((inst.arg >> 9) & 0xff) as usize;
                let c = ((inst.arg >> 17) & 0xff) as usize;
                let ra = register[a + base].as_ref().unwrap().clone();
                let nargs = b - 1;
                let nresults = c - 1;
                let ra = if let LuaValue::Closure(func) = ra {
                    func
                } else {
                    panic!("register[{}] is not function, actually {:?}", a, ra);
                };
                let lms = run_function(
                    &ra,
                    LuaMachineStatus { register, upvalues },
                    base + a,
                    nargs,
                    nresults,
                );
                register = lms.register;
                upvalues = lms.upvalues;
            }
            LuaOp::MOVE => {
                let a = (inst.arg & 0xff) as usize;
                let b = ((inst.arg >> 9) & 0xff) as usize;
                let rb = register[b + base].as_ref().unwrap().clone();
                register[a + base] = Some(rb);
            }
            LuaOp::LOADK => {
                let a = (inst.arg & 0xff) as usize;
                let bx = (inst.arg >> 8) as usize;
                let bx = &func.constants[bx];
                match bx.t {
                    LuaVariables::VSHRSTR => {
                        if let LuaValue::String(str) = &bx.value {
                            register[a + base] = Some(LuaValue::String(str.clone()));
                        } else {
                            panic!("Unsupported Type: {:?}", bx);
                        }
                    }
                    _ => {
                        panic!("Unsupported Type: {:?}", bx)
                    }
                }
            }
            LuaOp::LOADNIL => {
                let a = (inst.arg & 0xff) as usize;
                let b = ((inst.arg >> 9) & 0xff) as usize;
                for i in 0..b {
                    register[a + base + i] = Some(LuaValue::Nil());
                }
            }
            LuaOp::LOADI => {
                let a = (inst.arg & 0xff) as usize;
                let sbx = ((inst.arg >> 8) as i64) - 0xffff;
                register[a + base] = Some(LuaValue::Integer(sbx));
            }
            LuaOp::GETTABUP => {
                let a = (inst.arg & 0xff) as usize;
                let b = ((inst.arg >> 9) & 0xff) as usize;
                let c = ((inst.arg >> 17) & 0xff) as usize;
                let upval = &upvalues[b];
                let kc = func.constants[c].value.clone();
                let kc = if let LuaValue::String(value) = kc {
                    value
                } else {
                    panic!("key must be string!");
                };
                register[a + base] = Some(
                    upval
                        .get(&kc)
                        .unwrap_or_else(|| panic!("key[{}] not found from upvalue[{}]", kc, b))
                        .clone(),
                );
            }
            LuaOp::SETTABUP => {
                let a = (inst.arg & 0xff) as usize;
                let upval = &mut upvalues[a];
                let is_k = (inst.arg >> 8) & 0x01 != 0;
                let b = ((inst.arg >> 9) & 0xff) as usize;
                let c = ((inst.arg >> 17) & 0xff) as usize;
                let key = func.constants[b].value.clone();
                let key = if let LuaValue::String(value) = key {
                    value
                } else {
                    panic!("key must be string!");
                };
                let rc: &LuaValue = if is_k {
                    &func.constants[c].value
                } else {
                    register[c + base].as_ref().unwrap()
                };
                let rc = rc.clone();
                upval.insert(key, rc);
            }
            LuaOp::RETURN => {
                let a = (inst.arg & 0xff) as usize;
                let b = ((inst.arg >> 9) & 0xff) as usize;
                let nresults = b - 1;

                for i in 0..nresults {
                    register[base + i - 1] = Some(register[a + base + i].as_ref().unwrap().clone());
                }

                return LuaMachineStatus { register, upvalues };
            }
            LuaOp::RETURN1 => {
                let a = (inst.arg & 0xff) as usize;
                let ra = register[a + base].as_ref().unwrap().clone();
                register[base - 1] = Some(ra);
                return LuaMachineStatus { register, upvalues };
            }
            LuaOp::CLOSURE => {
                let a = (inst.arg & 0xff) as usize;
                let bx = (inst.arg >> 8) as usize;
                let p = Function::LuaFunction(func.protos[bx].clone());
                register[a + base] = Some(LuaValue::Closure(p));
            }
            LuaOp::FORLOOP => {
                let count = if let &LuaValue::Integer(x) = get_ra(&register, base + 1, inst.arg) {
                    x
                } else {
                    panic!("count: Must be signed integer!")
                };
                if count > 0 {
                    let step = if let &LuaValue::Integer(x) = get_ra(&register, base + 2, inst.arg)
                    {
                        x
                    } else {
                        panic!("step: Must be signed integer!")
                    };
                    let mut i = if let &LuaValue::Integer(x) = get_ra(&register, base, inst.arg) {
                        x
                    } else {
                        panic!("i: Must be signed integer!")
                    };

                    let bx = get_bx(inst.arg);

                    set_ra(
                        &mut register,
                        base + 1,
                        inst.arg,
                        Some(LuaValue::Integer(count - 1)),
                    );

                    i += step;

                    set_ra(&mut register, base, inst.arg, Some(LuaValue::Integer(i)));
                    set_ra(
                        &mut register,
                        base + 3,
                        inst.arg,
                        Some(LuaValue::Integer(i)),
                    );
                    idx -= bx;
                }
            }
            LuaOp::NEWTABLE => {
                // let ra = get_ra(&register, base, inst.arg);
                let b = get_b(inst.arg);
                let c = get_c(inst.arg);

                let b = if b > 0 { 1 << (b - 1) } else { b };
                let c = if test_k(inst.arg) {
                    c + get_ax(inst.arg) * (0x100)
                } else {
                    c
                };

                set_ra(
                    &mut register,
                    base,
                    inst.arg,
                    Some(LuaValue::Table(HashMap::new())),
                );
                idx += 1;
            }
            LuaOp::SETFIELD => {
                let ra = get_ra(&register, base, inst.arg);
                let b = get_b(inst.arg);
                let kb = &func.constants.get(b).unwrap().value;
                let c = get_c(inst.arg);
                let rkc = if test_k(inst.arg) {
                    func.constants.get(c).unwrap().value.clone()
                } else {
                    get_rc(&register, base, inst.arg).clone()
                };
                let key = if let LuaValue::String(s) = &kb {
                    s.clone()
                } else {
                    panic!("key must be string!")
                };

                let mut t = if let LuaValue::Table(t) = ra {
                    t.clone()
                } else {
                    panic!("only table supported!")
                };
                t.insert(key, rkc);
                set_ra(&mut register, base, inst.arg, Some(LuaValue::Table(t)));
            }
            LuaOp::GETFIELD => {
                let rb = get_rb(&register, base, inst.arg);
                let c = get_c(inst.arg);
                let kc = &func.constants.get(c).unwrap().value;

                let key = if let LuaValue::String(str) = kc {
                    str
                } else {
                    panic!("must be string: {:?}!", kc);
                };

                let v = if let LuaValue::Table(t) = rb {
                    t.get(key).unwrap().clone()
                } else {
                    panic!("must be table: {:?}!", kc);
                };

                set_ra(&mut register, base, inst.arg, Some(v));
            }
            LuaOp::FORPREP => {
                let init = if let &LuaValue::Integer(x) = get_ra(&register, base, inst.arg) {
                    x
                } else {
                    panic!("init: Must be signed integer!")
                };
                let limit = if let &LuaValue::Integer(x) = get_ra(&register, base + 1, inst.arg) {
                    x
                } else {
                    panic!("limit: Must be signed integer!")
                };
                let step = if let &LuaValue::Integer(x) = get_ra(&register, base + 2, inst.arg) {
                    x
                } else {
                    panic!("step: Must be signed integer!")
                };

                set_ra(
                    &mut register,
                    base + 3,
                    inst.arg,
                    Some(LuaValue::Integer(init)),
                );
                set_ra(
                    &mut register,
                    base + 1,
                    inst.arg,
                    Some(LuaValue::Integer(limit - init)),
                );
            }
            _ => panic!("Unsupported type: {:?}", inst.op),
        }

        idx += 1;
    }

    LuaMachineStatus { register, upvalues }
}

fn get_a(arg: u32) -> usize {
    (arg & 0xff) as usize
}

fn get_ax(arg: u32) -> usize {
    arg as usize
}

fn get_b(arg: u32) -> usize {
    ((arg >> 9) & 0xff) as usize
}

fn test_k(arg: u32) -> bool {
    ((arg >> 8) & 0x01) != 0
}

fn get_c(arg: u32) -> usize {
    ((arg >> 17) & 0xff) as usize
}

fn get_bx(arg: u32) -> usize {
    (arg >> 8) as usize
}

fn get_ra<'a>(register: &'a [Option<LuaValue>], base: usize, arg: u32) -> &'a LuaValue {
    let a = get_a(arg);
    register[a + base].as_ref().unwrap()
}

fn set_ra(register: &mut [Option<LuaValue>], base: usize, arg: u32, v: Option<LuaValue>) {
    let a = get_a(arg);
    register[a + base] = v;
}

fn get_rb<'a>(register: &'a [Option<LuaValue>], base: usize, arg: u32) -> &'a LuaValue {
    let b = get_b(arg);
    register[b + base].as_ref().unwrap()
}

fn get_rc<'a>(register: &'a [Option<LuaValue>], base: usize, arg: u32) -> &'a LuaValue {
    let c = get_c(arg);
    register[c + base].as_ref().unwrap()
}

fn run_function(
    func: &Function,
    lms: LuaMachineStatus,
    base: usize,
    nargs: usize,
    nresults: usize,
) -> LuaMachineStatus {
    match func {
        Function::LuaFunction(func) => run_lua_function(func, lms, base, nargs, nresults),
        Function::CFunction(func) => {
            let cfunc = func.func;
            cfunc(lms, base, nargs, nresults)
        }
    }
}

fn load_functions(
    bytecodes: &[u8],
    pos: &mut usize,
    lm: &LuaMachine,
    source: Option<String>,
) -> LuaFunction {
    let source = if let Some(source) = source {
        if let Some(str) = load_string(bytecodes, pos) {
            str
        } else {
            source
        }
    } else {
        load_string(bytecodes, pos).unwrap()
    };
    let line_defined = load_signed(bytecodes, pos);
    let last_line_defined = load_signed(bytecodes, pos);
    let num_params = load_u8(bytecodes, pos);
    let is_var_arg = load_u8(bytecodes, pos);
    let max_stack_size = load_u8(bytecodes, pos);
    let code_size = load_unsigned(bytecodes, pos);
    let code = load_bytes(bytecodes, pos, code_size * lm.instruction_size);
    let code = (0..code_size)
        .map(|i| i * 4)
        .map(|i| u32::from_le_bytes(code[i..i + 4].try_into().unwrap()))
        .collect::<Vec<_>>();
    let code = code
        .iter()
        .map(|&x| {
            let op: LuaOp = LuaOp::from_u8((x & 0x7f) as u8).unwrap();
            let arg = x >> 7;
            Instruction { op, arg }
        })
        .collect::<Vec<_>>();

    let constants: Vec<_> = (0..load_signed(bytecodes, pos))
        .map(|i| {
            let t = LuaVariables::from_u8(load_u8(bytecodes, pos)).unwrap();
            match t {
                LuaVariables::VSHRSTR | LuaVariables::VLNGSTR => {
                    let str = load_string(bytecodes, pos).unwrap();
                    let value = LuaValue::String(str);
                    Constant { t, value }
                }
                LuaVariables::VNUMFLT => {
                    let flt = load_f64(bytecodes, pos);
                    let value = LuaValue::Float(flt);
                    Constant { t, value }
                }
                LuaVariables::VNUMINT => {
                    let int = load_i64(bytecodes, pos);
                    let value = LuaValue::Integer(int);
                    Constant { t, value }
                }
                _ => {
                    panic!("Unknown type[{}]: {:?}", i, t)
                }
            }
        })
        .collect();

    // Load upvalues
    let upvalues: Vec<_> = (0..load_signed(bytecodes, pos))
        .map(|_| {
            let instack = load_u8(bytecodes, pos);
            let idx = load_u8(bytecodes, pos);
            let kind = load_u8(bytecodes, pos);
            UpValue { instack, idx, kind }
        })
        .collect();
    // load protos
    let protos: Vec<_> = (0..load_signed(bytecodes, pos))
        .map(|_| load_functions(bytecodes, pos, lm, Some(source.clone())))
        .collect();

    // load_debug
    let size = load_unsigned(bytecodes, pos);
    let line_info = load_bytes(bytecodes, pos, size).to_vec();
    let abs_line_info: Vec<_> = (0..load_signed(bytecodes, pos))
        .map(|_| {
            let pc = load_signed(bytecodes, pos);
            let line = load_signed(bytecodes, pos);
            AbsLineInfo { pc, line }
        })
        .collect();

    let loc_vars: Vec<_> = (0..load_signed(bytecodes, pos))
        .map(|_| {
            let var_name = load_string(bytecodes, pos).unwrap_or_default();
            let start_pc = load_signed(bytecodes, pos);
            let end_pc = load_signed(bytecodes, pos);
            LocVar {
                var_name,
                end_pc,
                start_pc,
            }
        })
        .collect();

    let upvalue_names: Vec<_> = (0..load_signed(bytecodes, pos))
        .map(|_| load_string(bytecodes, pos).unwrap_or_default())
        .collect();

    LuaFunction {
        is_var_arg,
        last_line_defined,
        line_defined,
        max_stack_size,
        num_params,
        source,
        code,
        constants,
        protos,
        upvalues,
        line_info,
        abs_line_info,
        loc_vars,
        upvalue_names,
    }
}

fn load_bytes<'a>(bytes: &'a [u8], pos: &mut usize, size: usize) -> &'a [u8] {
    let p = *pos;
    let x = &bytes[p..p + size];
    *pos += size;
    x
}

fn load_string(bytes: &[u8], pos: &mut usize) -> Option<String> {
    let size = load_unsigned(bytes, pos);
    if size == 0 {
        return None;
    }
    let bytes = load_bytes(bytes, pos, size - 1);
    return Some(String::from_utf8(bytes.to_vec()).unwrap());
}

fn load_u8(bytes: &[u8], pos: &mut usize) -> u8 {
    let x = bytes[*pos];
    *pos += 1;
    x
}

fn load_f64(bytes: &[u8], pos: &mut usize) -> f64 {
    let p = *pos;
    let x = f64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
    *pos += 8;
    x
}

fn load_i64(bytes: &[u8], pos: &mut usize) -> i64 {
    let p = *pos;
    let x = i64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
    *pos += 8;
    x
}

fn load_signed(bytes: &[u8], pos: &mut usize) -> i64 {
    return load_unsigned(bytes, pos) as i64;
}

fn load_unsigned(bytes: &[u8], pos: &mut usize) -> usize {
    let mut x: usize = 0;
    let mut b: usize;

    let limit = usize::MAX >> 7;
    loop {
        b = load_u8(bytes, pos) as usize;
        if x >= limit {
            panic!("integer overflow");
        }

        x = (x << 7) | (b & 0x7f);
        if (b & 0x80) != 0 {
            break;
        }
    }

    x
}
