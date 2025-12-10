from dataclasses import dataclass
from typing import Any, Optional
import tempfile
import logging
import os
import subprocess
import json

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
    location: Location

@dataclass
class CounterExample:
    values: list[VarValue]
    clause: 'ClauseInfo'

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
        cls(
            clause_assertions=[ClauseInfo.from_json(clause) for clause in data['clause_assertions']]
        )
    
    def clause_info_for_loc(self, loc: Loc) -> Optional[ClauseInfo]:
        return next(filter(lambda info: info.assert_loc == loc, self.clause_assertions))

QUANTIFIER_ITERATIONS = 128

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
    # TODO: handle arrays
    vars = {}

    for var in data:
        name = var['name']
        vars[name] = VarValue(
            var_name=name,
            # TODO: select proper representation based on signed or unsigned
            value=int(var['val-decimal']),
            location=Location(
                line=int(var['loc']['line']),
                column=int(var['loc']['col']),
            ),
        )
    
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
            line=location['line'],
            col=location['col'],
        )

        clause_info = metadata.clause_info_for_loc(location)
        if clause_info is None:
            # no matching clause found
            # TODO: maybe still report to LLM?
            continue

        vars = parse_counterexample_vars(metadata, counterexample['counter-example'])
        out.append(CounterExample(
            vars=vars,
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
        
        return parse_crux_results(metadata, results)

if __name__ == '__main__':
    gen_counterexamples('/tmp/testing.rs')
