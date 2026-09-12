//! Exact rational arithmetic for the theory solver.
//!
//! The whole point of this app is that a verdict is *derived*, not guessed, so the
//! arithmetic underneath it must not drift. `f64` would: `0.1 + 0.2 <= 0.3` is false
//! in binary floating point, and a policy that says "the discount may not exceed
//! 0.3" would then reject a perfectly legal 0.1 + 0.2. Every coefficient is therefore
//! a reduced `i128/i128` fraction and every operation is checked.
//!
//! Overflow is not a panic and not a wraparound — it is [`None`]. Callers propagate
//! that up as `Unknown`, which the verdict layer surfaces as `too_complex`. A number
//! we could not represent must never look like a proof.

use std::cmp::Ordering;
use std::fmt;

/// An exact rational. Invariant: `den > 0` and `gcd(|num|, den) == 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rat {
    num: i128,
    den: i128,
}

const fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    if a < 0 {
        -a
    } else {
        a
    }
}

impl Rat {
    pub const ZERO: Rat = Rat { num: 0, den: 1 };
    pub const ONE: Rat = Rat { num: 1, den: 1 };

    /// Build a reduced rational. `den == 0` is a caller bug and yields `None`, as
    /// does a numerator/denominator pair that cannot be normalized in `i128`.
    pub fn new(num: i128, den: i128) -> Option<Rat> {
        if den == 0 {
            return None;
        }
        let sign = if den < 0 { -1 } else { 1 };
        let g = gcd(num, den).max(1);
        let num = num.checked_div(g)?.checked_mul(sign)?;
        let den = den.checked_div(g)?.checked_abs()?;
        Some(Rat { num, den })
    }

    pub const fn int(v: i64) -> Rat {
        Rat {
            num: v as i128,
            den: 1,
        }
    }

    pub const fn numer(self) -> i128 {
        self.num
    }

    pub const fn denom(self) -> i128 {
        self.den
    }

    pub const fn is_zero(self) -> bool {
        self.num == 0
    }

    pub const fn is_negative(self) -> bool {
        self.num < 0
    }

    pub const fn is_positive(self) -> bool {
        self.num > 0
    }

    pub const fn is_integer(self) -> bool {
        self.den == 1
    }

    pub fn add(self, other: Rat) -> Option<Rat> {
        let n = self
            .num
            .checked_mul(other.den)?
            .checked_add(other.num.checked_mul(self.den)?)?;
        Rat::new(n, self.den.checked_mul(other.den)?)
    }

    pub fn sub(self, other: Rat) -> Option<Rat> {
        self.add(other.neg()?)
    }

    pub fn neg(self) -> Option<Rat> {
        Some(Rat {
            num: self.num.checked_neg()?,
            den: self.den,
        })
    }

    pub fn mul(self, other: Rat) -> Option<Rat> {
        Rat::new(
            self.num.checked_mul(other.num)?,
            self.den.checked_mul(other.den)?,
        )
    }

    pub fn div(self, other: Rat) -> Option<Rat> {
        if other.is_zero() {
            return None;
        }
        Rat::new(
            self.num.checked_mul(other.den)?,
            self.den.checked_mul(other.num)?,
        )
    }

    /// Largest integer `<= self`.
    pub fn floor(self) -> Option<i128> {
        let q = self.num.checked_div(self.den)?;
        if self.num % self.den != 0 && self.num < 0 {
            q.checked_sub(1)
        } else {
            Some(q)
        }
    }

    /// Smallest integer `>= self`.
    pub fn ceil(self) -> Option<i128> {
        let q = self.num.checked_div(self.den)?;
        if self.num % self.den != 0 && self.num > 0 {
            q.checked_add(1)
        } else {
            Some(q)
        }
    }

    /// Midpoint of two rationals, used to pick a witness value strictly between a
    /// lower and an upper bound.
    pub fn midpoint(self, other: Rat) -> Option<Rat> {
        self.add(other)?.div(Rat::int(2))
    }

    /// A short decimal rendering when the value is exactly representable, otherwise
    /// the fraction. Only ever used for display.
    pub fn to_display(self) -> String {
        if self.den == 1 {
            return self.num.to_string();
        }
        // Try an exact short decimal (den is 2^a * 5^b).
        let mut d = self.den;
        let mut scale = 0u32;
        while d % 2 == 0 && scale < 9 {
            d /= 2;
            scale += 1;
        }
        let mut d2 = d;
        let mut scale2 = 0u32;
        while d2 % 5 == 0 && scale2 < 9 {
            d2 /= 5;
            scale2 += 1;
        }
        if d2 == 1 {
            let places = scale.max(scale2).min(9);
            if let Some(pow) = 10i128.checked_pow(places) {
                if let Some(scaled) = self.num.checked_mul(pow) {
                    if scaled % self.den == 0 {
                        let v = scaled / self.den;
                        let sign = if v < 0 { "-" } else { "" };
                        let v = v.abs();
                        let unit = 10i128.pow(places);
                        let whole = v / unit;
                        let frac = v % unit;
                        if frac == 0 {
                            return format!("{sign}{whole}");
                        }
                        let frac = format!("{frac:0width$}", width = places as usize);
                        let frac = frac.trim_end_matches('0');
                        return format!("{sign}{whole}.{frac}");
                    }
                }
            }
        }
        format!("{}/{}", self.num, self.den)
    }
}

impl Default for Rat {
    /// Zero — the additive identity a [`crate::logic::Linear`] starts from.
    fn default() -> Self {
        Rat::ZERO
    }
}

impl PartialOrd for Rat {
    fn partial_cmp(&self, other: &Rat) -> Option<Ordering> {
        // den > 0 for both, so cross-multiplication preserves the direction.
        let lhs = self.num.checked_mul(other.den)?;
        let rhs = other.num.checked_mul(self.den)?;
        Some(lhs.cmp(&rhs))
    }
}

impl fmt::Display for Rat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduces_and_normalizes_sign() {
        let r = Rat::new(4, -8).unwrap();
        assert_eq!(r.numer(), -1);
        assert_eq!(r.denom(), 2);
    }

    #[test]
    fn decimal_addition_is_exact_where_floats_are_not() {
        // The float trap this module exists to avoid: 0.1 + 0.2 > 0.3 in f64.
        let tenth = Rat::new(1, 10).unwrap();
        let fifth = Rat::new(2, 10).unwrap();
        let three_tenths = Rat::new(3, 10).unwrap();
        assert_eq!(tenth.add(fifth).unwrap(), three_tenths);
        assert!(tenth.add(fifth).unwrap() <= three_tenths);
    }

    #[test]
    fn floor_and_ceil_round_toward_minus_and_plus_infinity() {
        assert_eq!(Rat::new(7, 2).unwrap().floor().unwrap(), 3);
        assert_eq!(Rat::new(7, 2).unwrap().ceil().unwrap(), 4);
        assert_eq!(Rat::new(-7, 2).unwrap().floor().unwrap(), -4);
        assert_eq!(Rat::new(-7, 2).unwrap().ceil().unwrap(), -3);
    }

    #[test]
    fn overflow_is_none_not_wraparound() {
        let huge = Rat::new(i128::MAX, 1).unwrap();
        assert!(huge.add(huge).is_none());
        assert!(huge.mul(huge).is_none());
    }

    #[test]
    fn display_prefers_short_decimals() {
        assert_eq!(Rat::new(1, 4).unwrap().to_display(), "0.25");
        assert_eq!(Rat::new(5, 1).unwrap().to_display(), "5");
        assert_eq!(Rat::new(1, 3).unwrap().to_display(), "1/3");
    }
}
