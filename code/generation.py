# Copyright (c) Microsoft Corporation. #
# Licensed under the MIT license.      #

import os
from pathlib import Path
from infer import LLM
from houdini import houdini
from refinement import Refinement
from veval import VEval, EvalScore
from utils import evaluate, code_change_is_safe, clean_code, get_nonlinear_lines
import subprocess
import os


class Generation:
    def __init__(
        self, config, logger, phase1_examples=["3", "6", "7"], repair_uniform=False
    ):
        self.config = config
        self.llm = LLM(config, logger)
        self.logger = logger
        self.default_refine_funcs = [
            self.constantrefine_inference,
            self.arraylen_inference,
            self.arrayrefine_inference,
            self.condlooprefine_inference,
        ]
        self.hdn = houdini(config)
        self.refinement = Refinement(config, logger)
        self.phase1_examples = phase1_examples
        self.repair_uniform = repair_uniform

        self.logger.info(
            "Generation initialized with phase1_examples: %s", self.phase1_examples
        )
        self.logger.info(
            "Generation initialized with repair_uniform: %s", self.repair_uniform
        )

    def direct_full_inference_with_smt2(
        self,
        verus_code,
        smt2_output,
        temp=0,
        answer_num=1,
        error="",
        use_simple=True,
        use_misc_examples=True,
    ):
        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        complex_instruction = """Your missions are to
1. Add loop invariants to the given Rust code, if there are loops in the code, so that Verus can verify the give function behaves exact what is described in the specifications
2. Add the proof blocks that could help Verus to prove the following code snippet. You need to analyze which locations in the code need to be proved and add the proof blocks to help Verus to prove the correctness of the code. You can insert multiple proof blocks in the code as long as they are necessary to prove the correctness of the code. You can also include new ghost variables that could help you to prove the correctness of the code.

Here are some principles that you have to follow:
Respond with the Rust code only, do not include any explanation.
If a function is marked with unimplemented!(), please leave it there and do NOT try to add new implementation.
You should never change or delete any existing code.
If this function contains no loop, feel free to leave it as it is without adding anything.

Please follow these steps in adding loop invariants for every loop:
1. You should identify every variable that is read in the loop  (e.g., x[k], y), particularly for array elements like x[k], and add an invariant about the initial value for EACH such variable and array;
2. You should identify every variable that is written (e.g., y = ..., x.set(..,..)) in every loop, and add an invariant about the value of that variable. Even if an invariant is already specified earlier in the program, please do repeat it in every loop suitable. Copy them in the response.
3. You should fully utilize the spec functions and proof functions in the invariant.

Here are some common locations where you can add proof blocks:
1. In the beginning of the function
2. Before the loop
3. In the beginning of the loop
4. In the end of the loop
5. Before the key operations
6. After the key operations

The proof block looks like this:
```
proof {
    // your proof code here
    // assert(...)
    // LEMMA_FUNCTION(...)
    // ...
} // Added by AI
```
The ghost variable looks like this:
```
let ghost ...; // Added by AI
```

If there is nothing to add for a function, that is OK. 
"""
        simple_instruction = """Please generate loop invariants and proof blocks for the given Rust code, so that Verus can verify the give function behaves exact what is described in the specifications. 

        For the reference, we have attached the smt2 output generated from SeaHorn (which you can parse from here https://github.com/seahorn/seahorn), use that as a context to generate the correct Verus specifications.

        Respond with the Verus code only, do not include any explanation.
"""

        if use_simple:
            self.logger.warning("Using simple instruction ...")
            instruction = simple_instruction
        else:
            self.logger.warning("Using complex instruction ...")
            instruction = complex_instruction

        examples = []
        if use_misc_examples:
            for f in sorted(
                os.listdir(os.path.join(self.config.example_path, "input-temp"))
            ):
                if f.endswith(".rs") and f.startswith("ex"):
                    input_file = os.path.join(self.config.example_path, "input-temp", f)
                    output_file = os.path.join(
                        self.config.example_path, "output-temp", f
                    )
                    input_content = open(input_file).read()
                    output_content = open(output_file).read()
                    examples.append({"query": input_content, "answer": output_content})
        else:
            for f in sorted(
                os.listdir(os.path.join(self.config.example_path, "input"))
            ):
                if f.endswith(".rs") and f[2] in self.phase1_examples:
                    input_file = os.path.join(self.config.example_path, "input", f)
                    output_file = os.path.join(self.config.example_path, "output", f)
                    input_content = open(input_file).read()
                    output_content = open(output_file).read()
                    examples.append({"query": input_content, "answer": output_content})
        with open("example.log", "w") as f:
            for ex in examples:
                f.write(ex["query"] + "\n")
                f.write(ex["answer"] + "\n\n")

        self.logger.info("Direct Full Inference ...")

        code_with_smt2 = verus_code + "\n===============================\n Here start the .smt2 output generated form seahorn.\n" + smt2_output
 
        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code_with_smt2,
            system,
            answer_num=answer_num,
            max_tokens=self.config.max_token,
            temp=temp,
        )
    
    # This long prompt is used in the alternative design where proof generation is done in one shot
    # without further phases of refinement or repair
    def direct_full_inference(
        self,
        code,
        temp=0,
        answer_num=1,
        error="",
        use_simple=True,
        use_misc_examples=True,
    ):
        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        complex_instruction = """Your missions are to
1. Add loop invariants to the given Rust code, if there are loops in the code, so that Verus can verify the give function behaves exact what is described in the specifications
2. Add the proof blocks that could help Verus to prove the following code snippet. You need to analyze which locations in the code need to be proved and add the proof blocks to help Verus to prove the correctness of the code. You can insert multiple proof blocks in the code as long as they are necessary to prove the correctness of the code. You can also include new ghost variables that could help you to prove the correctness of the code.

Here are some principles that you have to follow:
Respond with the Rust code only, do not include any explanation.
If a function is marked with unimplemented!(), please leave it there and do NOT try to add new implementation.
You should never change or delete any existing code.
If this function contains no loop, feel free to leave it as it is without adding anything.

Please follow these steps in adding loop invariants for every loop:
1. You should identify every variable that is read in the loop  (e.g., x[k], y), particularly for array elements like x[k], and add an invariant about the initial value for EACH such variable and array;
2. You should identify every variable that is written (e.g., y = ..., x.set(..,..)) in every loop, and add an invariant about the value of that variable. Even if an invariant is already specified earlier in the program, please do repeat it in every loop suitable. Copy them in the response.
3. You should fully utilize the spec functions and proof functions in the invariant.

Here are some common locations where you can add proof blocks:
1. In the beginning of the function
2. Before the loop
3. In the beginning of the loop
4. In the end of the loop
5. Before the key operations
6. After the key operations

The proof block looks like this:
```
proof {
    // your proof code here
    // assert(...)
    // LEMMA_FUNCTION(...)
    // ...
} // Added by AI
```
The ghost variable looks like this:
```
let ghost ...; // Added by AI
```

If there is nothing to add for a function, that is OK. 
"""
        simple_instruction = """Please generate loop invariants and proof blocks for the given Rust code, so that Verus can verify the give function behaves exact what is described in the specifications. 

Respond with the Rust code only, do not include any explanation.
"""

        if use_simple:
            self.logger.warning("Using simple instruction ...")
            instruction = simple_instruction
        else:
            self.logger.warning("Using complex instruction ...")
            instruction = complex_instruction

        examples = []
        if use_misc_examples:
            for f in sorted(
                os.listdir(os.path.join(self.config.example_path, "input-temp"))
            ):
                if f.endswith(".rs") and f.startswith("ex"):
                    input_file = os.path.join(self.config.example_path, "input-temp", f)
                    output_file = os.path.join(
                        self.config.example_path, "output-temp", f
                    )
                    input_content = open(input_file).read()
                    output_content = open(output_file).read()
                    examples.append({"query": input_content, "answer": output_content})
        else:
            for f in sorted(
                os.listdir(os.path.join(self.config.example_path, "input"))
            ):
                if f.endswith(".rs") and f[2] in self.phase1_examples:
                    input_file = os.path.join(self.config.example_path, "input", f)
                    output_file = os.path.join(self.config.example_path, "output", f)
                    input_content = open(input_file).read()
                    output_content = open(output_file).read()
                    examples.append({"query": input_content, "answer": output_content})
        with open("example.log", "w") as f:
            for ex in examples:
                f.write(ex["query"] + "\n")
                f.write(ex["answer"] + "\n\n")

        self.logger.info("Direct Full Inference ...")
        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=answer_num,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    def direct_inference_one_shot(self, code, temp=0, answer_num=1, error=""):
        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """Your mission is to add loop invariants to the given Rust code, if there are loops in the code, so that Verus can verify the give function behaves exact what is described in the specifications. 

Here are some principles that you have to follow:
Respond with Rust code only, do not include any explanation.
You should never change or delete existing Rust code.

First convert the rust code to an equivalent verus code having the correct syntax that is compilable. 
    a. You should be able to learn the syntax of Verus from the example added. 
    b. You can keep the main function empty as we want the specifications only for the function called in the main function in the Rust code. You can consider those invoked functions are the ones to be verified.

For example, for the following Rust code:

#![no_std]

#[no_mangle]
pub extern "C" fn count_to_n(n: u32) -> u32 {
    let mut i: u32 = 0;

    while i < n {
        i = i + 1;
    }

    i
}

#[no_mangle]
pub extern "C" fn main() -> u32 {
    count_to_n(5)
}

The corresponding Verus code is as follows:

use vstd::prelude::*;

verus! {

fn count_to_n(n: u32) -> (r: u32)
    ensures
        r == n,
{
    let mut i: u32 = 0;

    while i < n
        invariant
            i <= n,     // i stays within bounds
    {
        i = i + 1;
    }

    i
}

} // verus!

fn main() {}


Please follow these steps in adding loop invariants for every loop:
1. You should identify every variable that is read in the loop  (e.g., x[k], y), particularly for array elements like x[k], and add an invariant about the initial value for EACH such variable and array;
2. You should identify every variable that is written (e.g., y = ..., x.set(..,..)) in every loop, and add an invariant about the value of that variable. Even if an invariant is already specified earlier in the program, please do repeat it in every loop suitable.
3. You can leverage the spec functions and proof functions in the invariant.
"""
        # Integrate the Seq knowledge if needed
        instruction += self.refinement.add_seq_knowledge(code, instruction)

        examples = []

        for f in sorted(os.listdir(os.path.join(self.config.example_path, "input"))):
            if f.endswith(".rs") and f[2] in self.phase1_examples:
                input_file = os.path.join(self.config.example_path, "input", f)
                output_file = os.path.join(self.config.example_path, "output", f)
                input_content = open(input_file).read()
                output_content = open(output_file).read()
                examples.append({"query": input_content, "answer": output_content})

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=answer_num,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    # The default first-step of preliminary loop invariant generation
    def direct_inference(self, code, temp=0, answer_num=1, error=""):
        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """Your mission is to add loop invariants to the given Rust code, if there are loops in the code, so that Verus can verify the give function behaves exact what is described in the specifications. 

Here are some principles that you have to follow:
Respond with Rust code only, do not include any explanation.
You should never change or delete existing Rust code.

Please follow these steps in adding loop invariants for every loop:
1. You should identify every variable that is read in the loop  (e.g., x[k], y), particularly for array elements like x[k], and add an invariant about the initial value for EACH such variable and array;
2. You should identify every variable that is written (e.g., y = ..., x.set(..,..)) in every loop, and add an invariant about the value of that variable. Even if an invariant is already specified earlier in the program, please do repeat it in every loop suitable.
3. You can leverage the spec functions and proof functions in the invariant.
"""
        # Integrate the Seq knowledge if needed
        instruction += self.refinement.add_seq_knowledge(code, instruction)

        examples = []

        for f in sorted(os.listdir(os.path.join(self.config.example_path, "input"))):
            if f.endswith(".rs") and f[2] in self.phase1_examples:
                input_file = os.path.join(self.config.example_path, "input", f)
                output_file = os.path.join(self.config.example_path, "output", f)
                input_content = open(input_file).read()
                output_content = open(output_file).read()
                examples.append({"query": input_content, "answer": output_content})

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=answer_num,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    # The default first-step of preliminary loop invariant generation
    def direct_inference_with_smt2(self, code, smt2_content="", temp=0, answer_num=1, error=""):
        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """Your mission is to add method contracts (requires and ensures clauses) and loop invariants to the given Rust code, if there are loops in the code, so that Verus can verify the give function behaves exact what is described in the specifications. 

Here are some principles that you have to follow:
Respond with Verus code only, do not include any explanation.
You should never change or delete existing Rust code.


Please follow these steps in adding loop invariants for every loop:
1. You should identify every variable that is read in the loop  (e.g., x[k], y), particularly for array elements like x[k], and add an invariant about the initial value for EACH such variable and array;
2. You should identify every variable that is written (e.g., y = ..., x.set(..,..)) in every loop, and add an invariant about the value of that variable. Even if an invariant is already specified earlier in the program, please do repeat it in every loop suitable.
3. You can leverage the spec functions and proof functions in the invariant.
"""
        # Integrate the Seq knowledge if needed
        instruction += self.refinement.add_seq_knowledge(code, instruction)

        examples = []

        for f in sorted(os.listdir(os.path.join(self.config.example_path, "input"))):
            if f.endswith(".rs") and f[2] in self.phase1_examples:
                input_file = os.path.join(self.config.example_path, "input", f)
                output_file = os.path.join(self.config.example_path, "output", f)
                input_content = open(input_file).read()
                output_content = open(output_file).read()
                examples.append({"query": input_content, "answer": output_content})

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=answer_num,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    # This is an alternative design where the generation phase and refinement phase are combined into one prompt
    def direct_inference_with_refinement(self, code, temp=0, answer_num=1, error=""):
        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """
## Step 1: Add Loop Invariants
Your mission is to add loop invariants to the given Rust code, if there are loops in the code, so that Verus can verify the give function behaves exact what is described in the specifications.

Here are some principles that you have to follow:
Respond with Rust code only, do not include any explanation.
You should never change or delete existing Rust code.

Please follow these steps in adding loop invariants for every loop:
1. You should identify every variable that is read in the loop  (e.g., x[k], y), particularly for array elements like x[k], and add an invariant about the initial value for EACH such variable and array;
2. You should identify every variable that is written (e.g., y = ..., x.set(..,..)) in every loop, and add an invariant about the value of that variable. Even if an invariant is already specified earlier in the program, please do repeat it in every loop suitable.
3. You can leverage the spec functions and proof functions in the invariant.

## Step 2: Constant propagation refinement

If an upper bound or a lower bound about a constant function parameter (e.g., X < ..., X > ...) is provided in the function pre-condition (i.e., in the `requires' code block at the beginning of the function), 
please copy that (e.g., X < 10, X > 5) as a loop invariant to every loop in the function. 
Even if an invariant is already specified earlier in the program, please do repeat it in every loop suitable.

## Step 3: Array length refinement

For every loop in the function, please identify every array that is read (e.g., x[k]) or written (e.g., x.set(..,..)) in it, and then add a loop invariant that specifies the length of the array (i.e., x.len() == ...).

## Step 4: Quantifier range refinement

Please take the following steps to check every loop invariant that involves an array (e.g., x[k]) in the given Rust code:
If this array x[k] has been modified in this loop through x.set(), leave this invariant as it is, do NOT make any changes, and move on to the next invariant. 
Otherwise, when there is no x.set() in the loop, please make sure that the invariant covers every element in the array and hence has the form like `forall |k:int| 0<= k < x.len() ==> whatever-property'. When you make this change, please use a comment to explain why you believe the related array is never changed in the loop. Do NOT make any other changes to the code or the loop invariant!

## Step 5: Conditional loop invariant refinement

Your mission is to refine some loop invariants in the given Rust code only if the loop has special handling for the first iteration. This is what you should do: if an existing loop invariant P holds for all iterations of the loop except for the first iteration (e.g., some variable updates may only (not) occur during the first loop iteration), please leave P as it is and add another loop invariant conditioned on the loop index (e.g., index > 0 ==> P), following the example below. 
Do not change P or any other loop invariants in any other way."""

        self.logger.warning("Direct Inference unified with Refinement ...")

        # Integrate the Seq knowledge if needed
        instruction += self.refinement.add_seq_knowledge(code, instruction)

        examples = []

        for f in sorted(os.listdir(os.path.join(self.config.example_path, "input"))):
            if f.endswith(".rs") and f[2] in self.phase1_examples:
                input_file = os.path.join(self.config.example_path, "input", f)
                output_file = os.path.join(self.config.example_path, "output", f)
                input_content = open(input_file).read()
                output_content = open(output_file).read()
                examples.append({"query": input_content, "answer": output_content})

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=answer_num,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    #############################################
    ###The next few are the refinement agents####
    #############################################

    def arraylen_inference(self, code, temp=0, answer_num=1, error=""):
        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """
        For every loop in the function, please identify every array that is read (e.g., x[k]) or written (e.g., x.set(..,..)) in it, and then add a loop invariant that specifies the length of the array (i.e., x.len() == ...).

Here are some principles that you have to follow:
 You should only response with Rust code, and not include any explanation. 
 You should not make any other changes to the program.
"""
        examples = []

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=answer_num,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    def condlooprefine_inference(self, code, temp=0, answer_num=1, error=""):
        """
        This one checks if any loop invariant should be made to be conditional on loop indx, particularly if the invariant holds for all but the first interation of the loop.

        In terms of error fixing:
        ** If Verus complains that an array-related loop invariant does not hold before the loop,
        we can try this refinement.
        """

        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """Your mission is to refine some loop invariants in the given Rust code only if the loop has special handling for the first iteration. This is what you should do: if an existing loop invariant P holds for all iterations of the loop except for the first iteration (e.g., some variable updates may only (not) occur during the first loop iteration), please leave P as it is and add another loop invariant conditioned on the loop index (e.g., index > 0 ==> P), following the example below. 
Do not change P or any other loop invariants in any other way. """

        examples = []

        for f in sorted(
            os.listdir(os.path.join(self.config.example_path, "input-condinv"))
        ):
            if f.endswith(".rs"):
                input_file = os.path.join(self.config.example_path, "input-condinv", f)
                output_file = os.path.join(
                    self.config.example_path, "output-condinv", f
                )
                input_content = open(input_file).read()
                output_content = open(output_file).read()
                examples.append({"query": input_content, "answer": output_content})

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=1,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    # quantifier refinement
    def arrayrefine_inference(self, code, temp=0, answer_num=1, error=""):
        """
        This one checks if an array-related loop invariant has the right range clause:
        if the array was not modified in the loop, the range clause should be 0<= .. <array.len()
        otherwise, the range clause should be 0<= .. <i or i<= .. <array.len()

        In terms of error fixing:
        ** If Verus complains that an array-related loop invariant does not hold after the loop,
        we can check whether this array is actually not modified and hence should use [0, array.len) clause.
        or if this array is actually modified and hence should NOT use [0, array.len) clause.
        """

        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """Please take the following steps to check every loop invariant that involves an array (e.g., x[k]) in the given Rust code:
        If this array x[k] has been modified in this loop through x.set(), leave this invariant as it is, do NOT make any changes, and move on to the next invariant. 
        Otherwise, when there is no x.set() in the loop, please make sure that the invariant covers every element in the array and hence has the form like `forall |k:int| 0<= k < x.len() ==> whatever-property'. When you make this change, please use a comment to explain why you believe the related array is never changed in the loop. Do NOT make any other changes to the code or the loop invariant!

You should only response with Rust code, and not include any explanation.
You should NEVER ever add new variables, NEVER!
You should only make changes to existing loop invariants in the following ways, and you should not make any other changes to the program.
"""
        examples = []

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=1,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    def constantrefine_inference(self, code, temp=0, answer_num=1, error=""):
        """
        This one checks if any constant parameter related invariant is missing.

        In terms of error fixing:
        ** If Verus complains about arithmetic overflow,
        we can run this refinement.
        """

        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """
If an upper bound or a lower bound about a constant function parameter (e.g., X < ..., X > ...) is provided in the function pre-condition (i.e., in the `requires' code block at the beginning of the function), 
please copy that (e.g., X < 10, X > 5) as a loop invariant to every loop in the function. 
Even if an invariant is already specified earlier in the program, please do repeat it in every loop suitable.

Here are some principles that you have to follow:
 You should only response with Rust code, and not include any explanation. 
 You should not make any other changes to the program.
"""

        examples = []

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=1,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    def nonlinear_inference(self, code, temp=0, answer_num=1, error=""):
        """
        This one checks if any loop invariant is related to a non-linear property.

        In terms of error fixing:
        ** If any invariant is non-linear ...
        """

        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """Your mission is to add assert statements into the given Rust function to help Verus prove non-linear properties.

Here are some principles that you have to follow:
Response with the Rust code only, do not include any explanation.
You should only add assertions with non-linear property if necessary in the following ways, and you should not make any other changes to the program.

Nonlinear arithmetic involves equations that multiply, divide, or take the remainder of integer variables (e.g., x * (y * z) == (x * y) * z). Verus can reason about nonlinear arithmetic, but it needs to be told when to do so. To do this, you need to add a special keyword `nonlinear_arith' to the assert statement.
For example, if we know that variable X equals k*k+2*k and that variable Y equals (k+1)*(k+1), to prove that X+1 equals Y, we have to write the following statement to help Verus:

    assert(X+1 == Y) by (nonlinear_arith)
        requires
            X == k*k+2*k,
            Y == (k+1)*(k+1),
            0 < k,
            k < N,
            N <= 300,
            {}

In this example, the `nonlinear_arith' would enable Verus to use its non-linear reasoning to prove X+1 equals Y. The requires statements should include all the information that is needed to reason about the assert statement, including any variable bound information that is need to prove no-arithmetic overflow.

Please check the given program, and add nonlinear_arith assertion when Verus needs to reason about non-linear properties."""

        examples = []

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=1,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    def nonlbound_inference(self, code, temp=0, answer_num=1, error=""):
        """
        This one is to add bound for any nonlinear expressions.

        In terms of error fixing:
        ** arithmetic overflow
        """

        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """Your mission is to add assertions with `nonlinear_arith' keyword in the given Rust function to help Verus prove there is no arithmetic overflow for any non-linear expressions.

Here are some principles that you have to follow:
Response with the Rust code only, do not include any explanation.
You should only add assertions with non-linear property in the following ways, and you should not make any other changes to the program. Do not delete any existing assertions.

Verus cannot prove that a non-linear expression does not overflow unless you tell it the range of the expression.
For example, if a non-linear expression x*x*x is used in the program, only tell Verus 0 <= x <= 10 is not enough, we have to write the following statement to help Verus prove no arithmetic overflow for x*x*x:

    assert(0 < x*x*x <= 10 * 10 * 10) by (nonlinear_arith)
        requires
            0 < x,
            x <= 10,
            {}

In this example, the `nonlinear_arith' keyword enables Verus to use its non-linear reasoning, and 
the `requires' statements should include all the variable bound information needed to prove no-arithmetic overflow.

Please check the given program, and add above nonlinear_arith assertions when needed. Note that both the lower bound and upper bound of the expression should be specified in the assertion.
"""

        examples = []

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=1,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    # This refinement agent is deprecated as loop-isolation false can largely solve break's problem
    def breakloop_inference(self, code, temp=0, answer_num=1, error=""):
        """
        This one should be applied to loops that have early breaks
        """

        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """The break keyword serves as another way to prematurely exit a loop, but it carries a slight complication in terms of loop specifications. Unlike simple while loops whose loop conditions must only be false upon exiting, loops with a break command can exit regardless of whether the loop condition is true or not. Code including break commands are expected to explicitly specify post-loop conditions using ensures clause. Take a look at the example below about how to add `ensures` clause for a loop with break, and then add `ensures' clause for any loop that contains break in the provided code accordingly. If a loop does not have break in it, please do NOT make any changes.

You should only response with Rust code, and not include any explanation.
"""
        examples = self.refinement.get_examples("loopbreak")

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=1,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    def inference_with_smt2(self, code, smt2, temp=0, answer_num=1, error=""):
        system = "You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."

        instruction = """
        Now, please refine the current specs that you have already generated based on the smt2 output generated from SeaHorn (attached). 
        Please improve the current specs so that Verus can verify the given function behaves exactly as described in the specifications. 

 You should only response with Rust code, and not include any explanation. 
 You should not make any other changes to the program.
 Make sure that the final code you output can be compiled by Verus.
"""
        examples = []

        new_code_with_smt2 = code + "\n===============================\n Here start the .smt2 output generated form seahorn.\n" + smt2

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            new_code_with_smt2,
            system,
            answer_num=answer_num,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    # WARNING: This repair agent is **deprecated** (for now).
    # LLM is indeed capable of generating proof blocks, but doing it without error-guidance is not effective
    def proof_block_inference(self, code, temp=0, answer_num=1, error=""):
        system = """You are an experienced formal language programmer. You are very familiar with Verus, which is a tool for verifying the correctness of code written in Rust."""

        instruction = """Please add proof blocks to the given Rust code to help Verus prove the correctness of the code. You need to analyze which locations in the code need to be proved and add the proof blocks to help Verus to prove the correctness of the code. You can insert multiple proof blocks in the code as long as they are necessary to prove the correctness of the code. You can also include new ghost variables that could help you to prove the correctness of the code.

Here are some common locations where you can add proof blocks:
1. In the beginning of the function
2. Before the loop
3. In the beginning of the loop
4. In the end of the loop
5. Before the key operations
6. After the key operations

The proof block looks like this:
```
proof {
    // your proof code here
    // assert(...)
    // LEMMA_FUNCTION(...)
    // ...
} // Added by AI
```

The ghost variable looks like this:
```
let ghost ...; // Added by AI
```

Here are some principles that you have to follow:
 You should only response with Rust code, and not include any explanation."""
        examples = []
        for f in sorted(
            os.listdir(os.path.join(self.config.example_path, "input-proof"))
        ):
            if f.endswith(".rs"):
                input_file = os.path.join(self.config.example_path, "input-proof", f)
                output_file = os.path.join(self.config.example_path, "output-proof", f)
                input_content = open(input_file).read()
                output_content = open(output_file).read()
                examples.append({"query": input_content, "answer": output_content})
        with open("proof_block_example.txt", "w") as f:
            f.write("Query:\n" + examples[0]["query"])
            f.write("\n\nAnswer:\n" + examples[0]["answer"])

        return self.llm.infer_llm(
            self.config.aoai_generation_model,
            instruction,
            examples,
            code,
            system,
            answer_num=answer_num,
            max_tokens=self.config.max_token,
            temp=temp,
        )

    def generate_baseline_static_analyses(self, verus_code, seahorn_output, retry=25):
        """
        Generate the proof code.
        """
        temp = 1.0
        answer_num = 5

        best_code_of_all = verus_code
        best_score_of_all = EvalScore.get_worst_score()
        # Just send the seahorn code once to the LLM so that is saves in the env/context.
        # Compile corresponding seahorn compatiable rust code with seahorn -- some funciton call 
        # Returns -- > string of the .smt2 file. 

        # We promt the LLM with the (rustc, .smt2) file contents once. Just to initialize the context
        # 

        for i in range(retry):
            self.logger.info("Direct inference with baseline attempt %d" % i)
            candidates = self.direct_full_inference_with_smt2(verus_code, seahorn_output,  temp, answer_num)
            for cand_code in candidates:
                cand_code = clean_code(cand_code)
                veval = VEval(cand_code, self.logger)
                score = veval.eval_and_get_score()
                if score.is_correct():
                    return cand_code
                if score > best_score_of_all:
                    best_score_of_all = score
                    best_code_of_all = cand_code
        return best_code_of_all

    # An alternative design where the proof is generated in one step (no refinement or repair)
    def generate_baseline(self, code, retry=25):
        """
        Generate the proof code.
        """
        temp = 1.0
        answer_num = 5

        best_code_of_all = code
        best_score_of_all = EvalScore.get_worst_score()
        # Just send the seahorn code once to the LLM so that is saves in the env/context.
        # Compile corresponding seahorn compatiable rust code with seahorn -- some funciton call 
        # Returns -- > string of the .smt2 file. 

        # We promt the LLM with the (rustc, .smt2) file contents once. Just to initialize the context
        # 

        for i in range(retry):
            self.logger.info("Direct inference with baseline attempt %d" % i)
            candidates = self.direct_full_inference(code, temp, answer_num)
            for cand_code in candidates:
                cand_code = clean_code(cand_code)
                veval = VEval(cand_code, self.logger)
                score = veval.eval_and_get_score()
                if score.is_correct():
                    return cand_code
                if score > best_score_of_all:
                    best_score_of_all = score
                    best_code_of_all = cand_code
        return best_code_of_all

    def generate_with_proof_func(
        self,
        code,
        with_inference=True,
        with_refine=True,
        with_smt2=False,
        smt2_content="",
        learning_type=0,
        merge_cand=5,
        verbose=False,
        repair_steps=10,
        temp=1.0,
        temp_dir=Path("output-intermediate-temp"),
        disable_ranking=False,
    ):
        """
        Generate the proof code with the whole pipeline.
        This is the default pipeline for proof generation in AutoVerus.
        """
        temp_dir.mkdir(parents=True, exist_ok=True)
        answer_num = merge_cand
        original_code = code

        if with_inference:
            best_score_of_all = EvalScore.get_worst_score()
            best_score_of_valid = EvalScore.get_worst_score()
            code_pool = []
            best_code_of_all = original_code

            attempt = 0
            code_pool_stop_size = 4
            if disable_ranking:
                self.logger.warning("Disabled ranking")
                code_pool_stop_size = 1

            while attempt < 3:
                self.logger.info("Direct inference attempt {}".format(attempt))
                # Now use direct_inference.
                if learning_type == 0:
                    codes = self.direct_inference(
                        original_code, temp=temp, answer_num=answer_num
                    )
                else:
                    self.logger.info("From Rust to Verus: Running one shot learning approach ...")
                    codes = self.direct_inference_one_shot(
                        original_code, temp=temp, answer_num=answer_num
                    )
                found = False
                has_unsafe = False
                for i, cand_code in enumerate(codes):
                    self.logger.info(f"Checking candidate {attempt}-{i}")
                    cand_code = clean_code(cand_code)
                    newcode, _ = self.refinement.debug_type_error(cand_code)
                    if newcode:
                        cand_code = newcode

                    veval = VEval(cand_code, self.logger)
                    score = veval.eval_and_get_score()

                    is_safe_code_change = code_change_is_safe(
                        original_code, cand_code, self.config.verus_path, self.logger
                    )

                    if not is_safe_code_change:
                        has_unsafe = True

                    if score.is_correct() and is_safe_code_change:
                        self.logger.info("Verus succeeded!!")
                        return cand_code

                    if score > best_score_of_all:
                        best_score_of_all = score
                        best_code_of_all = cand_code

                    (temp_dir / f"{attempt}-{i}.rs").write_text(
                        cand_code
                        + "\n// is safe: "
                        + str(is_safe_code_change)
                        + "\n// Score: "
                        + str(score)
                    )
                    # TODO: We could try to fix the code with compilation error, instead of directly rejecting it
                    if (
                        "verus!" in cand_code
                        and is_safe_code_change
                        and not score.compilation_error
                    ):
                        found = True
                        self.logger.info(f"{attempt}-{i}.rs in code pool")
                        code_pool.append(cand_code)
                        if not (score < best_score_of_valid):
                            best_score_of_valid = score
                            self.logger.info(
                                f"{attempt}-{i}.rs is now the best proof candidate"
                            )
                            code = cand_code
                        if len(code_pool) >= code_pool_stop_size:
                            break

                if found and not has_unsafe:
                    break

                # if unsafe code was generated or if no valid code is fine at all,
                # better try another invocation to get more candidates
                self.logger.info("Regenerate...")
                attempt += 1

            if best_score_of_valid == EvalScore.get_worst_score():
                self.logger.error("No valid code found!")
                code = best_code_of_all
                code_pool = [code]
            else:
                # Try merging the valid codes to see if there is a better one.

                # Will also try an all-together merge, which may or may not be helpful
                allmerged_code = code
                for i, cp in enumerate(code_pool):
                    self.logger.info(f"Working on merge-{i}.rs")
                    try:
                        merged_code = self.hdn.merge_invariant(code, cp)
                        allmerged_code = self.hdn.merge_invariant(allmerged_code, cp)
                    except Exception as e:
                        self.logger.error(f"Error in merging code at step {i}: {e}")
                        continue

                    # selectively merged
                    veval = VEval(merged_code, self.logger)
                    score = veval.eval_and_get_score()
                    (temp_dir / f"merged-{i}.rs").write_text(
                        merged_code + "\n// Score: " + str(score)
                    )
                    if score.is_correct():
                        self.logger.info("Merged code is correct.")
                        return merged_code

                    if disable_ranking:
                        if not score.compilation_error:
                            self.logger.info(
                                "Disabled ranking and the code is not correct."
                            )
                            code = merged_code
                    elif not (score < best_score_of_valid):
                        self.logger.info("Merged code is better.")
                        best_score_of_valid = score
                        best_code_of_all = merged_code
                        # Only change the current code when the score isn't worse.
                        code = merged_code

                    self.logger.info(f"Running houdini on merge-{i}.rs")
                    hdn_failures, merge_code = self.hdn.run(merged_code)
                    if len(hdn_failures) == 0:
                        self.logger.info("Merged code with hdn is correct.")
                        return merge_code

                    # allmerged version
                    am_veval = VEval(allmerged_code, self.logger)
                    am_score = am_veval.eval_and_get_score()
                    (temp_dir / f"allmerged-{i}.rs").write_text(
                        allmerged_code + "\n// Score: " + str(am_score)
                    )
                    if am_score.is_correct():
                        self.logger.info("All merged code is correct.")
                        return allmerged_code
                    hdn_failures, hdnam_code = self.hdn.run(allmerged_code)
                    if len(hdn_failures) == 0:
                        self.logger.info("All merged code with hdn is correct.")
                        return hdnam_code

        # the best candidate is `code' now
        # score is cur_score
        veval = VEval(code, self.logger)
        cur_score = veval.eval_and_get_score()

        if with_refine:
            refine_funcs = self.default_refine_funcs
            if with_smt2:
                refine_funcs = [self.inference_with_smt2] + refine_funcs
            # If the code contains non-linear arithmetic
            nl_lines = get_nonlinear_lines(code, self.logger)
            if nl_lines:
                self.logger.warning("Non-linear arithmetic detected.")
                for _, _, text in nl_lines:
                    self.logger.warning(text)
                refine_funcs.append(self.nonlinear_inference)
                refine_funcs.append(self.nonlbound_inference)

            for i, func in enumerate(refine_funcs):
                self.logger.info("refining with %s" % func.__name__)
                attempt = 0
                original_code = code

                while attempt < 3:
                    # Only 1 refined candidate.
                    if func == self.inference_with_smt2:
                        code = func(
                            original_code,
                            smt2_content,
                            temp=temp,
                        )[0]
                    else:
                        code = func(original_code, temp=temp)[0]
                    # simple filtering
                    code = clean_code(code)
                    newcode = self.refinement.debug_type_error(code)[0]
                    if newcode:
                        code = newcode
                    if verbose:
                        self.logger.info(code)
                    if not code_change_is_safe(
                        original_code, code, self.config.verus_path, self.logger
                    ):
                        self.logger.info("Unsafe code change")
                        code = original_code
                    if "verus!" in code:
                        break

                    self.logger.info("regenerate...")
                    attempt += 1
                if code == original_code:
                    self.logger.info("Refinement did not change the code")
                    continue

                veval = VEval(code, self.logger)
                new_score = veval.eval_and_get_score()
                if new_score.is_correct():
                    self.logger.info("Verus succeeded with refinement!!")
                    return code

                hdn_failures, hdn_code = self.hdn.run(code)
                if len(hdn_failures) == 0:
                    self.logger.info("Verus succeeded with refinement and Houdini!")
                    return hdn_code

                # still errors left, let's see if we should accept the new version
                if func.__name__ == "condlooprefine_inference":
                    # condloop-refine is not often helpful, so we need to be cautious here
                    self.logger.info("New refined code under condloop is not better")
                    code = original_code
                elif disable_ranking:
                    if not score.compilation_error:
                        self.logger.info(
                            "Disabled ranking and the code is not correct."
                        )
                        code = original_code
                elif new_score.is_good_repair(cur_score):
                    # Now we use a loose condition to accept the new code.
                    self.logger.info("New refined code is a good repair")
                    self.logger.info(code)
                    cur_score = new_score
                    (temp_dir / f"refine-{i}.rs").write_text(code)
                else:
                    self.logger.info("New refined code is worse")
                    code = original_code

        if repair_steps > 0:
            (temp_dir / "before-repair.rs").write_text(
                code + "\n// Score: " + str(cur_score).replace("\n", " ")
            )
            repair_temp_dir = temp_dir / "repair"
            repair_temp_dir.mkdir(parents=True, exist_ok=True)

            if self.repair_uniform:
                # Ablation study: repair with uniform strategy
                code = self.refinement.repair_veval_in_one(
                    code, max_attempt=repair_steps, temp_dir=repair_temp_dir, temp=temp
                )
            else:
                code = self.refinement.repair_veval(
                    code, max_attempt=repair_steps, temp_dir=repair_temp_dir, temp=temp
                )

            veval = VEval(code, self.logger)
            score = veval.eval_and_get_score()
            if score.is_correct():
                self.logger.info("Verus succeeded after repair!!")
                return code

        (temp_dir / "final.rs").write_text(
            code + "\n// Score: " + str(score).replace("\n", " ")
        )

        # run houdini
        hdn_code = self.hdn.run(code)[1]
        hdn_veval = VEval(hdn_code, self.logger)
        hdn_score = hdn_veval.eval_and_get_score()
        (temp_dir / "final-hdn.rs").write_text(
            hdn_code + "\n// Score: " + str(hdn_score).replace("\n", " ")
        )
        if hdn_score.is_correct():
            self.logger.info("Verus succeeded with hdn!!")
            return hdn_code
        elif hdn_score > score:
            self.logger.info("Houdini code is better")
        else:
            self.logger.info("Original code is better")
        return code

    def execute_seahorn_and_get_smt2(self, input_file_name: str):
        """
        Runs seahorn_script.sh on the given Rust file.
        The input can be:
            - a bare filename ("foo")
            - a Rust filename ("foo.rs")
            - a full or relative path ("/x/y/foo.rs", "src/foo.rs")
        We extract the base filename without extension and use it for Seahorn.
        
        Calls seahorn_script.sh located in root_path,
        even if this Python file runs inside root_path/code.
        """

        root = self.config.root_path
        script_path = os.path.join(root, "seahorn_script.sh")

        # Extract file basename without extension
        file_basename = os.path.basename(input_file_name)
        file_key = file_basename[:-3] if file_basename.endswith(".rs") else file_basename

        # SMT2 output file path (absolute!)
        smt2_path = os.path.join(root, "smt_output", f"{file_key}.smt2")

        # Remove old file if exists
        if os.path.exists(smt2_path):
            os.remove(smt2_path)

        try:
            print(f"Running {script_path} {file_key} ...")

            # IMPORTANT: run from root directory
            result = subprocess.run(
                [script_path, file_key],
                cwd=root,                 # <-- FIX: run from correct directory
                capture_output=True,
                text=True
            )

            # Debug
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)

            # Check exit code
            if result.returncode != 0:
                print(f"Error: script exited with code {result.returncode}")
                return None

            # Check SMT2 file
            if not os.path.exists(smt2_path):
                print(f"SMT2 file not found: {smt2_path}")
                return None

            # Read SMT2 contents
            with open(smt2_path, "r") as f:
                return f.read()

        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def run(self, input_file, output_file, input_seahorn_file=None, args: dict = {}):
        baseline = args.get("is_baseline", False)
        repair_steps = args.get("repair", 5)
        merge_cand = args.get("merge", 5)
        baseline_with_seahorn = args.get("baseline_with_seahorn", False)
        with_smt2 = args.get("with_smt2", False)
        learning_type = args.get("learning_type", 0)
        temp = args.get("temp", 1.0)
        phase_uniform = args.get("phase_uniform", False)
        disable_ranking = args.get("disable_ranking", False)
        direct_repair = args.get("direct_repair", False)
        disable_one_refinement = args.get("disable_one_refinement", -1)

        self.logger.info("Using baseline?: %s" % str(baseline))

        if disable_one_refinement >= 0 and disable_one_refinement < len(
            self.default_refine_funcs
        ):
            self.logger.warning(
                "Disable one refinement function: %s"
                % self.default_refine_funcs[disable_one_refinement].__name__
            )
            self.default_refine_funcs = (
                self.default_refine_funcs[:disable_one_refinement]
                + self.default_refine_funcs[disable_one_refinement + 1 :]
            )

        content = open(input_file).read()
        # content_smt2 = open(input_seahorn_file)
        output_file = Path(output_file)
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(output_dir) / ("intermediate-" + output_file.stem)
        temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Generating proof code" + (" with baseline" if baseline else "")
        )
        self.logger.info("Temperature: " + str(temp))

        # Various alternate designs
        if baseline:
            self.logger.info("Generate with baseline mode")
            code = self.generate_baseline(content)
        elif baseline_with_seahorn:
            self.logger.info("Generate with baseline mode with Seahorn output")
            content_smt2 = self.execute_seahorn_and_get_smt2(input_file) 
            code = self.generate_baseline_static_analyses(content, content_smt2)
        elif phase_uniform:
            self.logger.info("Generate with uniform refinement mode")
            self.direct_inference = self.direct_inference_with_refinement
            code = self.generate_with_proof_func(
                content,
                with_refine=False,
                merge_cand=merge_cand,
                verbose=True,
                repair_steps=repair_steps,
                temp_dir=temp_dir,
                temp=temp,
                disable_ranking=disable_ranking,
            )
        elif direct_repair:
            self.logger.info("Generate with direct repair mode")
            code = self.generate_with_proof_func(
                content,
                with_inference=False,
                with_refine=False,
                merge_cand=merge_cand,
                verbose=True,
                repair_steps=repair_steps,
                temp_dir=temp_dir,
                temp=temp,
                disable_ranking=disable_ranking,
            )
        else:
            # default/recommended
            content_smt2 = ""
            if with_smt2:
                self.logger.info("Generate with refinement mode with Seahorn output")
                content_smt2 = self.execute_seahorn_and_get_smt2(input_file) 
            else:
                self.logger.info("Generate with refinement mode")
            code = self.generate_with_proof_func(
                content,
                with_refine=True,
                merge_cand=merge_cand,
                verbose=True,
                with_smt2=with_smt2,
                learning_type=learning_type,
                smt2_content=content_smt2,
                repair_steps=repair_steps,
                temp_dir=temp_dir,
                temp=temp,
                disable_ranking=disable_ranking,
            )

        score, _ = evaluate(code, self.config.verus_path)
        is_safe = code_change_is_safe(
            content, code, self.config.verus_path, self.logger, debug=False
        )
        code += "\n// Score: " + str(score)
        code += "\n// Safe: " + str(is_safe)

        with open(output_file, "w") as wf:
            wf.write(code)
        self.logger.info("finished!")
