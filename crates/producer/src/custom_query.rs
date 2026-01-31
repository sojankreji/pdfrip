use super::Producer;

enum ValueIterator {
    Numeric(Box<dyn Iterator<Item = usize>>),
    Alphabetic(Box<dyn Iterator<Item = String>>),
}

struct VariableBlock {
    content: String,
    is_pattern: bool,
}

pub struct CustomQuery {
    blocks: Vec<VariableBlock>,
    current_values: Vec<String>,
    iterators: Vec<ValueIterator>,
    iterator_sizes: Vec<usize>,
    current_indices: Vec<usize>,
    size: usize,
    exhausted: bool,
}

impl CustomQuery {
    pub fn new(query: &str, _add_preceding_zeros: bool) -> Self {
        let blocks = Self::parse_query_blocks(query);
        let mut iterators = Vec::new();
        let mut iterator_sizes = Vec::new();
        let mut current_values = Vec::new();
        let mut size = 1usize;

        // Create iterators for each pattern block
        for block in &blocks {
            if block.is_pattern {
                if let Some((chars, count)) = Self::parse_char_range(&block.content) {
                    // Character pattern
                    let iter_size = chars.len().pow(count as u32);
                    iterator_sizes.push(iter_size);
                    size = size.saturating_mul(iter_size);
                    let iterator = Self::create_char_iterator(chars, count);
                    iterators.push(ValueIterator::Alphabetic(Box::new(iterator)));
                    current_values.push(String::new());
                } else {
                    // Numeric pattern
                    let (start, end) = Self::parse_range(&block.content);
                    let iter_size = end - start;
                    iterator_sizes.push(iter_size);
                    size = size.saturating_mul(iter_size);
                    iterators.push(ValueIterator::Numeric(Box::new(start..end)));
                    current_values.push(String::new());
                }
            }
        }

        // Initialize current values with first values from each iterator
        let current_indices = vec![0; iterators.len()];

        Self {
            blocks,
            current_values,
            iterators,
            iterator_sizes,
            current_indices,
            size,
            exhausted: false,
        }
    }

    /// Parse query into blocks of literal text and patterns
    fn parse_query_blocks(query: &str) -> Vec<VariableBlock> {
        let mut blocks = Vec::new();
        let mut current_pos = 0;
        
        while current_pos < query.len() {
            if let Some(open_brace) = query[current_pos..].find('{') {
                let actual_open = current_pos + open_brace;
                
                // Add literal text before the brace
                if open_brace > 0 {
                    blocks.push(VariableBlock {
                        content: query[current_pos..actual_open].to_string(),
                        is_pattern: false,
                    });
                }
                
                // Find matching close brace
                if let Some(close_brace) = query[actual_open..].find('}') {
                    let actual_close = actual_open + close_brace;
                    let pattern = &query[actual_open + 1..actual_close];
                    
                    blocks.push(VariableBlock {
                        content: pattern.to_string(),
                        is_pattern: true,
                    });
                    
                    current_pos = actual_close + 1;
                } else {
                    // No matching close brace, treat rest as literal
                    blocks.push(VariableBlock {
                        content: query[actual_open..].to_string(),
                        is_pattern: false,
                    });
                    break;
                }
            } else {
                // No more braces, add rest as literal
                blocks.push(VariableBlock {
                    content: query[current_pos..].to_string(),
                    is_pattern: false,
                });
                break;
            }
        }
        
        blocks
    }

    fn parse_range(range: &str) -> (usize, usize) {
        let mut bounds = range.split('-').map(|n| n.parse::<usize>().unwrap());
        let start = bounds.next().unwrap();
        let end = bounds.next().unwrap();

        (start, end + 1)
    }

    /// Parse character range patterns like [A-Z]3, [a-z]2, [A-Za-z]4, [0-9]4
    /// Returns Some((characters_vec, count)) or None if not a char pattern
    fn parse_char_range(range: &str) -> Option<(Vec<char>, usize)> {
        // Check if it matches pattern like [A-Z]3
        if !range.starts_with('[') {
            return None;
        }

        let bracket_end = range.find(']')?;
        let char_spec = &range[1..bracket_end];
        let count_str = &range[bracket_end + 1..];
        let count = count_str.parse::<usize>().ok()?;

        let mut chars = Vec::new();

        // Handle simple cases manually for common patterns
        if char_spec == "A-Z" {
            for c in b'A'..=b'Z' {
                chars.push(c as char);
            }
        } else if char_spec == "a-z" {
            for c in b'a'..=b'z' {
                chars.push(c as char);
            }
        } else if char_spec == "0-9" {
            for c in b'0'..=b'9' {
                chars.push(c as char);
            }
        } else if char_spec == "A-Za-z" {
            // Add uppercase A-Z
            for c in b'A'..=b'Z' {
                chars.push(c as char);
            }
            // Add lowercase a-z
            for c in b'a'..=b'z' {
                chars.push(c as char);
            }
        } else if char_spec == "a-zA-Z" {
            // Add lowercase a-z
            for c in b'a'..=b'z' {
                chars.push(c as char);
            }
            // Add uppercase A-Z
            for c in b'A'..=b'Z' {
                chars.push(c as char);
            }
        } else if char_spec == "A-Za-z0-9" || char_spec == "a-zA-Z0-9" {
            // Alphanumeric
            for c in b'A'..=b'Z' {
                chars.push(c as char);
            }
            for c in b'a'..=b'z' {
                chars.push(c as char);
            }
            for c in b'0'..=b'9' {
                chars.push(c as char);
            }
        } else {
            // Generic parser for other patterns
            let parts: Vec<&str> = char_spec.split('-').collect();
            
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i].len() == 1 && parts[i + 1].len() == 1 {
                    // This is a simple range like A-Z or 0-9
                    let start_char = parts[i].chars().next().unwrap();
                    let end_char = parts[i + 1].chars().next().unwrap();
                    
                    for c in (start_char as u8)..=(end_char as u8) {
                        chars.push(c as char);
                    }
                    i += 2;
                } else {
                    // Single character
                    if let Some(c) = parts[i].chars().next() {
                        if !chars.contains(&c) {
                            chars.push(c);
                        }
                    }
                    i += 1;
                }
            }
        }

        Some((chars, count))
    }

    /// Create an iterator that generates all combinations of characters
    fn create_char_iterator(chars: Vec<char>, length: usize) -> impl Iterator<Item = String> {
        let total_combinations = chars.len().pow(length as u32);
        (0..total_combinations).map(move |mut idx| {
            let mut result = String::new();
            for _ in 0..length {
                let char_idx = idx % chars.len();
                result.push(chars[char_idx]);
                idx /= chars.len();
            }
            result.chars().rev().collect()
        })
    }
}

impl Producer for CustomQuery {
    fn next(&mut self) -> Result<Option<Vec<u8>>, String> {
        if self.exhausted {
            return Ok(None);
        }

        // On first call, populate current_values with first values
        if self.current_values.iter().all(|v| v.is_empty()) {
            for (idx, iter) in self.iterators.iter_mut().enumerate() {
                match iter {
                    ValueIterator::Numeric(num_iter) => {
                        if let Some(value) = num_iter.next() {
                            self.current_values[idx] = value.to_string();
                        } else {
                            self.exhausted = true;
                            return Ok(None);
                        }
                    }
                    ValueIterator::Alphabetic(char_iter) => {
                        if let Some(value) = char_iter.next() {
                            self.current_values[idx] = value;
                        } else {
                            self.exhausted = true;
                            return Ok(None);
                        }
                    }
                }
            }

            // Build and return first combination
            return Ok(Some(self.build_current_combination().into_bytes()));
        }

        // Increment to next combination (like a multi-digit counter)
        let mut carry = true;
        for idx in (0..self.iterators.len()).rev() {
            if !carry {
                break;
            }

            self.current_indices[idx] += 1;

            if self.current_indices[idx] < self.iterator_sizes[idx] {
                // Get next value from this iterator
                match &mut self.iterators[idx] {
                    ValueIterator::Numeric(num_iter) => {
                        if let Some(value) = num_iter.next() {
                            self.current_values[idx] = value.to_string();
                            carry = false;
                        }
                    }
                    ValueIterator::Alphabetic(char_iter) => {
                        if let Some(value) = char_iter.next() {
                            self.current_values[idx] = value;
                            carry = false;
                        }
                    }
                }
            } else {
                // Reset this iterator and carry to next position
                self.current_indices[idx] = 0;
                
                // Recreate the iterator for this position
                let block_idx = self.blocks.iter()
                    .enumerate()
                    .filter(|(_, b)| b.is_pattern)
                    .nth(idx)
                    .map(|(i, _)| i)
                    .unwrap();
                
                let block = &self.blocks[block_idx];
                
                if let Some((chars, count)) = Self::parse_char_range(&block.content) {
                    let iterator = Self::create_char_iterator(chars, count);
                    self.iterators[idx] = ValueIterator::Alphabetic(Box::new(iterator));
                    
                    // Get first value from recreated iterator
                    if let ValueIterator::Alphabetic(char_iter) = &mut self.iterators[idx] {
                        if let Some(value) = char_iter.next() {
                            self.current_values[idx] = value;
                        }
                    }
                } else {
                    let (start, end) = Self::parse_range(&block.content);
                    self.iterators[idx] = ValueIterator::Numeric(Box::new(start..end));
                    
                    // Get first value from recreated iterator
                    if let ValueIterator::Numeric(num_iter) = &mut self.iterators[idx] {
                        if let Some(value) = num_iter.next() {
                            self.current_values[idx] = value.to_string();
                        }
                    }
                }
                
                carry = true;
            }
        }

        if carry {
            // We've exhausted all combinations
            self.exhausted = true;
            return Ok(None);
        }

        Ok(Some(self.build_current_combination().into_bytes()))
    }

    fn size(&self) -> usize {
        self.size
    }
}

impl CustomQuery {
    fn build_current_combination(&self) -> String {
        let mut result = String::new();
        let mut pattern_idx = 0;
        
        for block in &self.blocks {
            if block.is_pattern {
                result.push_str(&self.current_values[pattern_idx]);
                pattern_idx += 1;
            } else {
                result.push_str(&block.content);
            }
        }
        
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uppercase_pattern() {
        let mut query = CustomQuery::new("prefix{[A-Z]2}suffix", false);
        
        // Should generate AA, AB, AC, ..., AZ, BA, BB, ..., ZZ (26*26 = 676 combinations)
        assert_eq!(query.size(), 676);
        
        println!("\n=== Testing [A-Z]2 pattern ===");
        println!("First 10 matches:");
        for i in 0..10 {
            if let Some(val) = query.next().unwrap() {
                println!("  {}: {}", i + 1, String::from_utf8(val).unwrap());
            }
        }
    }

    #[test]
    fn test_lowercase_pattern() {
        let mut query = CustomQuery::new("{[a-z]3}", false);
        
        // Should generate aaa, aab, ..., zzz (26^3 = 17576 combinations)
        assert_eq!(query.size(), 17576);
        
        println!("\n=== Testing [a-z]3 pattern ===");
        println!("First 15 matches:");
        for i in 0..15 {
            if let Some(val) = query.next().unwrap() {
                println!("  {}: {}", i + 1, String::from_utf8(val).unwrap());
            }
        }
    }

    #[test]
    fn test_mixed_case_pattern() {
        let mut query = CustomQuery::new("{[A-Za-z]2}", false);
        
        // Should generate all combinations of upper and lowercase (52^2 = 2704 combinations)
        assert_eq!(query.size(), 2704);
        
        println!("\n=== Testing [A-Za-z]2 pattern ===");
        println!("First 20 matches:");
        for i in 0..20 {
            if let Some(val) = query.next().unwrap() {
                println!("  {}: {}", i + 1, String::from_utf8(val).unwrap());
            }
        }
    }

    #[test]
    fn test_single_char_pattern() {
        let mut query = CustomQuery::new("test{[A-Z]1}.txt", false);
        
        // Should generate 26 combinations (A-Z)
        assert_eq!(query.size(), 26);
        
        println!("\n=== Testing [A-Z]1 pattern ===");
        println!("All 26 matches:");
        let mut count = 0;
        while let Some(val) = query.next().unwrap() {
            count += 1;
            print!("  {}", String::from_utf8(val).unwrap());
            if count % 6 == 0 { println!(); }
        }
        println!();
    }

    #[test]
    fn test_numeric_still_works() {
        let mut query = CustomQuery::new("doc{1-5}", false);
        
        // Should still work with numeric ranges
        assert_eq!(query.size(), 5);
        
        println!("\n=== Testing numeric range 1-5 ===");
        println!("All matches:");
        while let Some(val) = query.next().unwrap() {
            println!("  {}", String::from_utf8(val).unwrap());
        }
    }

    #[test]
    fn test_four_char_uppercase() {
        let mut query = CustomQuery::new("{[A-Z]4}", false);
        
        // Should generate AAAA through ZZZZ (26^4 = 456976 combinations)
        assert_eq!(query.size(), 456976);
        
        println!("\n=== Testing [A-Z]4 pattern ===");
        println!("First 25 matches:");
        for i in 0..25 {
            if let Some(val) = query.next().unwrap() {
                println!("  {}: {}", i + 1, String::from_utf8(val).unwrap());
            }
        }
    }

    #[test]
    fn test_numeric_char_pattern() {
        let mut query = CustomQuery::new("pin{[0-9]4}", false);
        
        // Should generate 0000 through 9999 (10^4 = 10000 combinations)
        assert_eq!(query.size(), 10000);
        
        println!("\n=== Testing [0-9]4 pattern ===");
        println!("First 20 matches:");
        for i in 0..20 {
            if let Some(val) = query.next().unwrap() {
                println!("  {}: {}", i + 1, String::from_utf8(val).unwrap());
            }
        }
    }

    #[test]
    fn test_alphanumeric_pattern() {
        let mut query = CustomQuery::new("{[A-Za-z0-9]2}", false);
        
        // Should generate all alphanumeric combinations (62^2 = 3844 combinations)
        assert_eq!(query.size(), 3844);
        
        println!("\n=== Testing [A-Za-z0-9]2 pattern ===");
        println!("First 30 matches:");
        for i in 0..30 {
            if let Some(val) = query.next().unwrap() {
                println!("  {}: {}", i + 1, String::from_utf8(val).unwrap());
            }
        }
    }

    #[test]
    fn test_multiple_variable_blocks() {
        let mut query = CustomQuery::new("{[A-Z]2}1477{[A-Z]1}", false);
        
        // Should generate all combinations: AA1477A, AA1477B, ..., ZZ1477Z
        // 26*26 * 26 = 17,576 combinations
        assert_eq!(query.size(), 17576);
        
        println!("\n=== Testing multiple blocks {{[A-Z]2}}1477{{[A-Z]1}} ===");
        println!("First 30 matches:");
        for i in 0..30 {
            if let Some(val) = query.next().unwrap() {
                println!("  {}: {}", i + 1, String::from_utf8(val).unwrap());
            }
        }
    }

    #[test]
    fn test_multiple_blocks_with_prefix_suffix() {
        let mut query = CustomQuery::new("DMCPR{[A-Z]5}1477{[A-Z]1}", false);
        
        // Should generate DMCPRAAAAA1477A through DMCPRZZZZZ1477Z
        // 26^5 * 26 = 308,915,776 combinations
        assert_eq!(query.size(), 308915776);
        
        println!("\n=== Testing DMCPR{{[A-Z]5}}1477{{[A-Z]1}} ===");
        println!("First 20 matches:");
        for i in 0..20 {
            if let Some(val) = query.next().unwrap() {
                println!("  {}: {}", i + 1, String::from_utf8(val).unwrap());
            }
        }
    }

    #[test]
    fn test_three_variable_blocks() {
        let mut query = CustomQuery::new("{[A-Z]1}-{[0-9]2}-{[a-z]1}", false);
        
        // Should generate A-00-a through Z-99-z
        // 26 * 100 * 26 = 67,600 combinations
        assert_eq!(query.size(), 67600);
        
        println!("\n=== Testing three blocks {{[A-Z]1}}-{{[0-9]2}}-{{[a-z]1}} ===");
        println!("First 30 matches:");
        for i in 0..30 {
            if let Some(val) = query.next().unwrap() {
                println!("  {}: {}", i + 1, String::from_utf8(val).unwrap());
            }
        }
    }

    #[test]
    fn test_verify_specific_combination() {
        let mut query = CustomQuery::new("DMCPR1477{[A-Z]1}", false);
        
        // Should include DMCPR1477K
        assert_eq!(query.size(), 26);
        
        let mut found_k = false;
        while let Some(val) = query.next().unwrap() {
            let s = String::from_utf8(val).unwrap();
            if s == "DMCPR1477K" {
                found_k = true;
                println!("\n✓ Found expected value: {}", s);
                break;
            }
        }
        
        assert!(found_k, "Should have found DMCPR1477K");
    }
}
