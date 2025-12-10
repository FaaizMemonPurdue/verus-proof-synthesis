from dataclasses import dataclass
from typing import Any, Optional
import tempfile
import logging
import os
import subprocess
import json
from enum import Enum

from lynette import lynette

logger = logging.getLogger(__name__)

@dataclass
class Location:
    line: int
    column: int

@dataclass
class VarValue:
    var_name: str
    value: Any
    # TODO: location
    # location: Location

@dataclass
class CounterExample:
    values: list[VarValue]
    clause: 'ClauseInfo'

    def to_prompt(self) -> str:
        vars = ''.join(f'{value.var_name} = {value.value}\n' for value in self.values)
        return f'Counterexample for clause: `{self.clause.clause}` on line {self.clause.clause_loc.line}, column {self.clause.clause_loc.col}\n{vars}'

def prompt_for_counterexamples(counterexamples: list[CounterExample]) -> str:
    counterexample_text = '\n'.join(counterexample.to_prompt() for counterexample in counterexamples)
    return f'''
Counterexamples for incorrect invariant and ensures clauses are shown below. The specific clause along with its line and column number in the original source code are displayed.
The lines after each clause indicate an assignment of inputs to the function which cause the clause to not hold.
These counterexamples may be useful to help you debug failing Verus specifications.

Counterexamples:
{counterexample_text}
'''

@dataclass
class Loc:
    line: int
    col: int

    @classmethod
    def from_json(cls, data):
        return cls(
            line=data['line'],
            col=data['col']
        )

@dataclass
class ClauseInfo:
    clause: str
    assert_loc: Loc
    clause_loc: Loc

    @classmethod
    def from_json(cls, data):
        return cls(
            clause=data['clause'],
            assert_loc=Loc.from_json(data['assert_loc']),
            clause_loc=Loc.from_json(data['clause_loc']),
        )

@dataclass
class Metadata:
    clause_assertions: list[ClauseInfo]

    @classmethod
    def from_json(cls, data):
        return cls(
            clause_assertions=[ClauseInfo.from_json(clause) for clause in data['clause_assertions']]
        )
    
    def clause_info_for_loc(self, loc: Loc) -> Optional[ClauseInfo]:
        return next(filter(lambda info: info.assert_loc == loc, self.clause_assertions), None)

class SymbolType(Enum):
    VALUE = 'Value'
    ARRAY = 'Array'
    ARRAY_LEN = 'ArrayLen'

@dataclass
class SymbolInfo:
    name: str
    type: SymbolType
    # only present for Value and Array
    type_name: Optional[str]
    # only present for array
    index: Optional[int]

    @classmethod
    def from_symbol_name(cls, symbol_name: str):
        assert symbol_name[0] == 'S'
        json_encoded = bytes.fromhex(symbol_name[1:]).decode('utf-8')
        data = json.loads(json_encoded)

        symbol_type = data['symbol_type']
        return cls(
            name=data['name'],
            type=SymbolType(symbol_type['type']),
            type_name=symbol_type.get('type_name'),
            index=symbol_type.get('index'),
        )
    
    # use appropriate value for var based on type
    def extract_int(self, var_data: dict) -> int:
        
        if self.type_name in ['u8', 'u16', 'u32', 'u64', 'u128', 'usize']:
            return parse_int_val(var_data['val-unsigned'])
        elif self.type_name in ['i8', 'i16', 'i32', 'i64', 'i128', 'isize']:
            return parse_int_val(var_data['val'])
        
        # non int not supported yet
        assert False

def parse_int_val(val: str) -> int:
    # for some reason random trailing letters are put on values by crux-mir
    # think it depends on bit width of bitvector
    while val[-1] not in 'abcdefABCDEF1234567890':
        val = val[:-1]
    return int(val, 16)

QUANTIFIER_ITERATIONS = 128
MAX_ARRAY_SIZE = 8

CARGO_TOML_CONTENTS = '''
[package]
name = "counterexample_gen"
version = "0.1.0"
authors = ["CS560 Group 3"]
edition = "2024"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]

[lib]
path = "lib.rs"
'''

def parse_counterexample_vars(metadata: Metadata, data) -> list[VarValue]:
    vars = {}

    for var in data:
        print(var)
        info = SymbolInfo.from_symbol_name(var['name'])
        name = info.name

        if info.type == SymbolType.VALUE:
            vars[name] = VarValue(
                var_name=name,
                value=info.extract_int(var),
            )
        else:
            if name not in vars:
                vars[name] = VarValue(
                    var_name=name,
                    value=[0] * MAX_ARRAY_SIZE,
                )

            var_val = vars[name]
            
            if info.type == SymbolType.ARRAY:
                # array may have been shrunk, don't want to cause oob exception
                if info.index < len(var_val.value):
                    var_val.value[info.index] = info.extract_int(var)
            elif info.type == SymbolType.ARRAY_LEN:
                length = parse_int_val(var['val-unsigned'])
                var_val.value = var_val.value[:length]
    
    return list(vars.values())

def parse_crux_results(metadata: Metadata, code_file: str, results) -> list[CounterExample]:
    assert type(results) is list
    out = []

    for counterexample in results:
        if counterexample['status'] != 'fail':
            # no counterexample found for a given clause
            continue

        location = counterexample['location']
        file = location['file']
        if file != code_file:
            # counterexample for some library, ignore
            # TODO: maybe still report to LLM?
            continue

        location = Loc(
            line=int(location['line']),
            col=int(location['col']),
        )

        clause_info = metadata.clause_info_for_loc(location)
        if clause_info is None:
            # no matching clause found
            # TODO: maybe still report to LLM?
            continue

        vars = parse_counterexample_vars(metadata, counterexample['counter-example'])
        out.append(CounterExample(
            values=vars,
            clause=clause_info,
        ))
    
    return out

def gen_counterexamples(file: str) -> list[CounterExample]:
    with tempfile.TemporaryDirectory() as tmpdir:
        crux_tests_file = os.path.join(tmpdir, 'lib.rs')
        metadata_file = os.path.join(tmpdir, 'metadata.json')
        result = lynette.assert_transform(file, crux_tests_file, metadata_file, True, QUANTIFIER_ITERATIONS)
        print(result)
        if result.returncode != 0:
            logger.error('assert transforming file failed')
            return []
        
        with open(metadata_file, 'r') as f:
            metadata = Metadata.from_json(json.load(f))
        
        cargo_toml_file = os.path.join(tmpdir, 'Cargo.toml')
        with open(cargo_toml_file, 'w') as f:
            f.write(CARGO_TOML_CONTENTS)
        
        out_dir = os.path.join(tmpdir, 'crux_out')
        
        # don't check return code, crux returns failure even when successful compilation, but counterexample found
        result = subprocess.run(
            f'cd {tmpdir} && cargo crux-test -- --iteration-bound={QUANTIFIER_ITERATIONS} --recursion-bound={QUANTIFIER_ITERATIONS} --output-directory={out_dir}',
            shell=True,
        )
        print(result)

        with open(os.path.join(out_dir, 'report.json'), 'r') as f:
            results = json.load(f)
        
        return parse_crux_results(metadata, crux_tests_file, results)

def gen_counterexamples_prompt(file: str) -> str:
    counterexamples = gen_counterexamples(file)
    return prompt_for_counterexamples(counterexamples)

if __name__ == '__main__':
    print(gen_counterexamples_prompt('/tmp/testing.rs'))
