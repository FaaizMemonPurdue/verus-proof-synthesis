use vstd::prelude::*;

verus! {

fn sum(mut input: &[u8]) -> (r: Vec<u8>)
ensures forall |i| 0 <= i < r.len() ==> r[i] == 69
{
    let mut output = Vec::new();
    let mut i = 0;
    let n = input.len();
    // while i < n {
    //     i += 1;
    //     output.push(69);
    // }
    output.push(70);
    output
}

} // verus!
