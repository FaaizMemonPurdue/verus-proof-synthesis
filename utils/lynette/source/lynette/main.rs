use vstd::prelude::*;

verus! {

fn inc(n: u8) -> (r: u8)
requires n < 255
ensures (n == 8) ==> (r == n + 1),
forall |n: int| n > 5
{
    if n == 5 {
        n + 3
    } else {
        n + 1
    }
}

} // verus!
