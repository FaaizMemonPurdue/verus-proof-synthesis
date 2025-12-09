use vstd::prelude::*;

verus! {

// fn inc(n: u8) -> (r: u8)
// ensures 0 < n < 20
// {
//     if n == 5 {
//         n + 3
//     } else {
//         n + 1
//     }
// }

fn sum(mut input: &[u8]) -> (r: Vec<u8>)
ensures forall |i| 0 <= i < r.len() ==> r[i] == 69
{
    let mut output = Vec::new();
    let mut i = 0;
    let n = input.len();
    while i < n {
        i += 1;
        output.push(69);
    }
    output
}

} // verus!
