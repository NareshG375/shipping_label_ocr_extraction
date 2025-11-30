"""
Utility functions for the OCR system
"""
import os
import json
import csv
from datetime import datetime
from typing import List, Dict
import numpy as np
from PIL import Image


def save_results_json(results: Dict, output_path: str):
    """
    Save results to JSON file
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_results_csv(results: List[Dict], output_path: str):
    """
    Save results to CSV file
    
    Args:
        results: List of result dictionaries
        output_path: Output file path
    """
    if not results:
        return
    
    keys = results[0].keys()
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Calculate accuracy of predictions
    
    Args:
        predictions: List of predicted texts
        ground_truth: List of ground truth texts
    
    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    if not predictions:
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions)


def calculate_partial_match_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Calculate accuracy allowing partial matches
    
    Args:
        predictions: List of predicted texts
        ground_truth: List of ground truth texts
    
    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    if not predictions:
        return 0.0
    
    correct = 0
    for pred, truth in zip(predictions, ground_truth):
        if pred and truth:
            # Consider it correct if prediction is contained in truth or vice versa
            if pred in truth or truth in pred:
                correct += 1
            # Or if they're very similar (Levenshtein distance)
            elif levenshtein_similarity(pred, truth) > 0.8:
                correct += 1
    
    return correct / len(predictions)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Edit distance as integer
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity ratio between two strings
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Similarity ratio between 0 and 1
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    distance = levenshtein_distance(s1, s2)
    return 1 - (distance / max_len)


def create_timestamp() -> str:
    """
    Create timestamp string
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: str):
    """
    Ensure directory exists
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_test_images(directory: str) -> List[str]:
    """
    Load all test images from directory
    
    Args:
        directory: Directory containing images
    
    Returns:
        List of image file paths
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(os.path.join(directory, filename))
    
    return sorted(image_files)


def resize_image_for_display(image: Image.Image, max_size: int = 800) -> Image.Image:
    """
    Resize image for display while maintaining aspect ratio
    
    Args:
        image: PIL Image
        max_size: Maximum dimension size
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    if width > height:
        if width > max_size:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            return image
    else:
        if height > max_size:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            return image
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def format_confidence(confidence: float) -> str:
    """
    Format confidence value as percentage
    
    Args:
        confidence: Confidence value (0-1)
    
    Returns:
        Formatted string
    """
    return f"{confidence * 100:.1f}%"


def generate_report(results: List[Dict]) -> Dict:
    """
    Generate summary report from results
    
    Args:
        results: List of processing results
    
    Returns:
        Report dictionary
    """
    report = {
        'total_images': len(results),
        'successful_extractions': sum(1 for r in results if r.get('extracted_text')),
        'failed_extractions': sum(1 for r in results if not r.get('extracted_text')),
        'average_confidence': 0.0,
        'timestamp': datetime.now().isoformat()
    }
    
    confidences = [r.get('confidence', 0) for r in results if r.get('confidence')]
    if confidences:
        report['average_confidence'] = sum(confidences) / len(confidences)
    
    if report['total_images'] > 0:
        report['success_rate'] = report['successful_extractions'] / report['total_images']
    else:
        report['success_rate'] = 0.0
    
    return report