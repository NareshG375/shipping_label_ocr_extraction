"""
Text extraction and pattern matching module - IMPROVED VERSION
"""
import re
from typing import List, Dict, Optional


class TextExtractor:
    """Extract specific patterns from OCR text"""
    
    def __init__(self):
        """Initialize text extractor"""
        # Pattern for _1_ format (flexible)
        #self.target_pattern = r'\w+_1_\w*'
        self.target_pattern = r'(\d+)\s*(\d+_1_\w+)'

        
    def clean_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors
        
        Args:
            text: Raw OCR text
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Common OCR mistakes
        replacements = {
            '_l_': '_1_',
            '_I_': '_1_',
        }
        
        cleaned = text
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    

    def extract_target_text(self, text: str) -> Optional[str]:

        if not text:
            return None

        cleaned = self.clean_ocr_errors(text)

        match = re.search(self.target_pattern, cleaned)
        if match:
            return match.group(1) + match.group(2)

        # Try original text
        match = re.search(self.target_pattern, text)
        if match:
            return match.group(1) + match.group(2)

        return None
        
    
    def extract_all_matches(self, text: str) -> List[str]:
        """
        Extract all texts containing _1_ pattern
        
        Args:
            text: Input text from OCR
        
        Returns:
            List of all matches
        """
        if not text:
            return []
        
        cleaned_text = self.clean_ocr_errors(text)
        matches = re.findall(self.target_pattern, cleaned_text, re.MULTILINE)
        return matches
    
    def extract_with_context(self, text: str, context_lines: int = 1) -> List[Dict]:
        """
        Extract pattern with surrounding context
        
        Args:
            text: Input text from OCR
            context_lines: Number of lines before/after to include
        
        Returns:
            List of dicts with match and context
        """
        if not text:
            return []
        
        cleaned_text = self.clean_ocr_errors(text)
        lines = cleaned_text.split('\n')
        results = []
        
        for i, line in enumerate(lines):
            match = re.search(self.target_pattern, line)
            if match:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                
                results.append({
                    'match': match.group(),
                    'line': line,
                    'line_number': i,
                    'context': '\n'.join(lines[start:end])
                })
        
        return results
    
    def clean_text(self, text: str) -> str:
        """
        Clean OCR text by removing noise
        
        Args:
            text: Raw OCR text
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might be OCR errors
        # but keep underscores and common punctuation
        text = re.sub(r'[^\w\s\-_:,.()/]', '', text)
        
        return text.strip()
    
    def validate_pattern(self, text: str) -> Dict:
        """
        Validate if extracted text matches expected pattern
        
        Args:
            text: Text to validate
        
        Returns:
            Dict with validation result and details
        """
        result = {
            'is_valid': False,
            'pattern': text,
            'issues': []
        }
        
        if not text:
            result['issues'].append('Empty text')
            return result
        
        # Check if matches pattern
        if not re.search(self.target_pattern, text):
            result['issues'].append('Does not match _1_ pattern')
            return result
        
        # Additional validation
        parts = text.split('_1_')
        if len(parts) != 2:
            result['issues'].append('Invalid structure')
            return result
        
        prefix, suffix = parts
        
        # Check prefix length (should be reasonable)
        if len(prefix) < 5:
            result['issues'].append('Prefix too short')
        elif len(prefix) > 50:
            result['issues'].append('Prefix too long')
        
        # Check suffix length
        if len(suffix) < 2:
            result['issues'].append('Suffix too short')
        elif len(suffix) > 20:
            result['issues'].append('Suffix too long')
        
        # If no issues found, mark as valid
        if not result['issues']:
            result['is_valid'] = True
        
        return result
    
    def find_similar_patterns(self, text: str) -> List[str]:
        """
        Find patterns similar to _1_ (like _2_, _3_, etc.)
        
        Args:
            text: Input text
        
        Returns:
            List of similar patterns found
        """
        if not text:
            return []
        
        cleaned_text = self.clean_ocr_errors(text)
        
        # Pattern for _N_ where N is any digit
        similar_pattern = r'[\w\d]+_\d+_[\w\d]+'
        matches = re.findall(similar_pattern, cleaned_text, re.MULTILINE)
        
        return list(set(matches))  # Remove duplicates
    
    def extract_line_containing_pattern(self, text: str) -> Optional[str]:
        """
        Extract the complete line containing the target pattern
        
        Args:
            text: Input text from OCR
        
        Returns:
            Complete line containing pattern or None
        """
        if not text:
            return None
        
        cleaned_text = self.clean_ocr_errors(text)
        lines = cleaned_text.split('\n')
        
        for line in lines:
            if re.search(self.target_pattern, line):
                return line.strip()
        
        return None