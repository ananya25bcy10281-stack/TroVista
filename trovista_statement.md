# Trovista Fashion Recommender System - Technical Statement

## Application Overview

**Application Name:** Trovista Fashion Recommendation System  
**Type:** Web Application (Flask-based)  
**Purpose:** AI-powered fashion recommendation engine  
**Deployment:** Local development server  
**Version:** 1.0

---

## Executive Summary

Trovista is a machine learning-powered fashion recommendation system that provides personalized clothing suggestions based on two input methods:
1. **Text-based recommendations** - Using body measurements and preferences
2. **Image-based recommendations** - Using uploaded fashion images

The system leverages K-Nearest Neighbors (KNN) algorithms with TF-IDF text vectorization and MobileNetV2 image embeddings to provide relevant fashion suggestions.

---

## System Architecture

### Technology Stack

**Backend Framework:**
- Flask (Python web framework)
- Debug mode enabled for development

**Machine Learning Libraries:**
- scikit-learn (KNN algorithms, NearestNeighbors)
- TensorFlow/Keras (MobileNetV2 for image processing)
- joblib (Model serialization)

**Data Processing:**
- pandas (Data manipulation and CSV handling)
- numpy (Numerical computations and array operations)

**Frontend:**
- HTML templates (Jinja2 templating engine)
- Flash messaging for user feedback

---

## Data Architecture

### Primary Dataset

**File:** `Myntra Fasion Clothing.csv`  
**Location:** `dataset/` directory  
**Loading:** Pandas CSV reader with `low_memory=False`

**Dataset Columns (Expected):**
- `productDisplayName` / `productName` - Product title
- `masterCategory` - Main category classification
- `subCategory` - Subcategory classification
- `articleType` - Specific article type
- `mrp` - Maximum retail price
- `discountedPr