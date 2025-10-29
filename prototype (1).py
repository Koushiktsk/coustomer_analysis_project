import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class DemographicPredictor:
    def __init__(self):

        self.age_weights = {
            'Headphones': {'under_25': 0.7, '25_to_40': 0.8, 'over_40': 0.6},
            'Earbuds': {'under_25': 0.8, '25_to_40': 0.7, 'over_40': 0.4},
            'Speakers': {'under_25': 0.6, '25_to_40': 0.7, 'over_40': 0.5},
            'Microphone': {'under_25': 0.5, '25_to_40': 0.8, 'over_40': 0.4},
            'Speaker Pads': {'under_25': 0.3, '25_to_40': 0.6, 'over_40': 0.9},
            'Ear Pads': {'under_25': 0.4, '25_to_40': 0.5, 'over_40': 0.3},
            'Earbuds Case': {'under_25': 0.7, '25_to_40': 0.5, 'over_40': 0.2}
        }


        self.gender_weights = {
            'Headphones': {'male': 0.6, 'female': 0.4},
            'Earbuds': {'male': 0.4, 'female': 0.6},
            'Speakers': {'male': 0.7, 'female': 0.3},
            'Microphone': {'male': 0.8, 'female': 0.2},
            'Speaker Pads': {'male': 0.8, 'female': 0.2},
            'Ear Pads': {'male': 0.5, 'female': 0.5},
            'Earbuds Case': {'male': 0.3, 'female': 0.7}
        }

        # Brand Gender Signals (additional modifiers)
        self.male_leaning_brands = ['JBL', 'Audio', 'MMT ACOUSTIX', 'boAt']
        self.female_leaning_brands = ['SOULWIT', 'Kapture', 'Apple', 'Samsung']

        # Price range mappings for age groups
        self.age_price_preferences = {
            'under_25': {'min': 0, 'max': 2000, 'weight': 1.2},
            '25_to_40': {'min': 1500, 'max': 8000, 'weight': 1.1},
            'over_40': {'min': 3000, 'max': float('inf'), 'weight': 1.3}
        }

    def preprocess_data(self, df):


        df = df[df['price'] >= 0].copy()
        df['date'] = pd.to_datetime(df['date'])


        df['simple_category'] = df['generic_name'].map({
            'Headphones': 'Headphones',
            'Earbuds': 'Earbuds',
            'Speaker Pads': 'Speaker Pads',
            'Microphone': 'Microphone',
            'Earbuds Case': 'Earbuds Case',
            'Ear Pads': 'Ear Pads'
        }).fillna('Speakers')

        return df

    def calculate_customer_features(self, customer_data):

        if len(customer_data) == 0:
            return None

        features = {}


        features['transaction_count'] = len(customer_data)
        features['total_spent'] = customer_data['price'].sum()
        features['avg_order_value'] = customer_data['price'].mean()
        features['unique_categories'] = customer_data['simple_category'].nunique()
        features['unique_brands'] = customer_data['brand'].nunique()


        category_dist = customer_data['simple_category'].value_counts(normalize=True)
        for category in self.age_weights.keys():
            features[f'category_share_{category}'] = category_dist.get(category, 0)


        brand_counts = customer_data['brand'].value_counts()
        features['male_brand_preference'] = sum([brand_counts.get(brand, 0)
                                                 for brand in self.male_leaning_brands]) / len(customer_data)
        features['female_brand_preference'] = sum([brand_counts.get(brand, 0)
                                                   for brand in self.female_leaning_brands]) / len(customer_data)


        features['price_min'] = customer_data['price'].min()
        features['price_max'] = customer_data['price'].max()
        features['price_std'] = customer_data['price'].std()


        if 'date' in customer_data.columns:
            date_range = (customer_data['date'].max() - customer_data['date'].min()).days
            features['purchase_span_days'] = max(1, date_range)
            features['purchase_frequency'] = len(customer_data) / max(1, date_range / 30.44)  # per month


            customer_data['day_of_week'] = customer_data['date'].dt.dayofweek
            weekend_purchases = len(customer_data[customer_data['day_of_week'].isin([5, 6])])
            features['weekend_shopper_ratio'] = weekend_purchases / len(customer_data)
        else:
            features['purchase_span_days'] = 1
            features['purchase_frequency'] = len(customer_data)
            features['weekend_shopper_ratio'] = 0.5

        return features

    def calculate_age_score(self, features, age_group):

        if not features:
            return 0

        score = 0
        total_weight = 0


        for category in self.age_weights.keys():
            category_share = features.get(f'category_share_{category}', 0)
            if category_share > 0:
                category_weight = self.age_weights[category][age_group]
                score += category_weight * category_share
                total_weight += category_share


        avg_price = features['avg_order_value']
        price_pref = self.age_price_preferences[age_group]
        if price_pref['min'] <= avg_price <= price_pref['max']:
            price_modifier = price_pref['weight']
        else:
            price_modifier = 0.8


        frequency = features['purchase_frequency']
        if age_group == 'under_25':
            freq_modifier = min(1.3, 1.0 + frequency * 0.1)  # Bonus for high frequency
        elif age_group == '25_to_40':
            freq_modifier = 1.1 if 1 <= frequency <= 4 else 0.9
        else:  # over_40
            freq_modifier = 1.2 if frequency < 2 and features['avg_order_value'] > 3000 else 0.9


        diversity_modifier = 1.0
        if age_group == 'over_40':
            diversity_modifier = min(1.2, 1.0 + features['unique_categories'] * 0.05)
        elif age_group == 'under_25':
            diversity_modifier = 1.1 if features['unique_categories'] <= 2 else 0.95

        final_score = score * price_modifier * freq_modifier * diversity_modifier
        return max(0, final_score)

    def calculate_gender_score(self, features, gender):

        if not features:
            return 0

        score = 0


        for category in self.gender_weights.keys():
            category_share = features.get(f'category_share_{category}', 0)
            if category_share > 0:
                category_weight = self.gender_weights[category][gender]
                score += category_weight * category_share


        if gender == 'male':
            brand_modifier = 1.0 + features['male_brand_preference'] * 0.3
        else:
            brand_modifier = 1.0 + features['female_brand_preference'] * 0.3


        weekend_modifier = 1.0
        if features['weekend_shopper_ratio'] > 0.6:
            weekend_modifier = 1.1  # Slight bonus for family-oriented shopping


        price_modifier = 1.0
        if gender == 'male' and features['avg_order_value'] > 4000:
            price_modifier = 1.2
        elif gender == 'female' and 1000 <= features['avg_order_value'] <= 3000:
            price_modifier = 1.1

        final_score = score * brand_modifier * weekend_modifier * price_modifier
        return max(0, final_score)

    def predict_demographics(self, customer_data):

        features = self.calculate_customer_features(customer_data)

        if not features:
            return {
                'age_prediction': 'unknown',
                'age_probabilities': {'under_25': 0.33, '25_to_40': 0.33, 'over_40': 0.33},
                'gender_prediction': 'unknown',
                'gender_probabilities': {'male': 0.5, 'female': 0.5},
                'confidence': 0.0
            }


        age_scores = {
            'under_25': self.calculate_age_score(features, 'under_25'),
            '25_to_40': self.calculate_age_score(features, '25_to_40'),
            'over_40': self.calculate_age_score(features, 'over_40')
        }


        gender_scores = {
            'male': self.calculate_gender_score(features, 'male'),
            'female': self.calculate_gender_score(features, 'female')
        }


        age_total = sum(age_scores.values())
        gender_total = sum(gender_scores.values())

        if age_total > 0:
            age_probabilities = {k: v / age_total for k, v in age_scores.items()}
        else:
            age_probabilities = {'under_25': 0.33, '25_to_40': 0.33, 'over_40': 0.33}

        if gender_total > 0:
            gender_probabilities = {k: v / gender_total for k, v in gender_scores.items()}
        else:
            gender_probabilities = {'male': 0.5, 'female': 0.5}


        age_prediction = max(age_probabilities, key=age_probabilities.get)
        gender_prediction = max(gender_probabilities, key=gender_probabilities.get)


        confidence = self._calculate_confidence(features, age_probabilities, gender_probabilities)

        return {
            'age_prediction': age_prediction,
            'age_probabilities': age_probabilities,
            'gender_prediction': gender_prediction,
            'gender_probabilities': gender_probabilities,
            'confidence': confidence,
            'features_used': features
        }

    def _calculate_confidence(self, features, age_probs, gender_probs):

        volume_factor = min(1.0, features['transaction_count'] * 0.15)


        age_strength = max(age_probs.values())
        gender_strength = max(gender_probs.values())
        signal_factor = (age_strength + gender_strength) / 2


        diversity_factor = min(1.0, features['unique_categories'] * 0.2)


        confidence = (volume_factor * 0.3 + signal_factor * 0.5 + diversity_factor * 0.2)

        return round(confidence, 3)

    def analyze_customer_base(self, df):

        df = self.preprocess_data(df)
        results = []

        print("Analyzing customer demographics...")
        customer_count = 0

        for user_id in df['user'].unique():
            customer_data = df[df['user'] == user_id]
            prediction = self.predict_demographics(customer_data)

            results.append({
                'user_id': user_id,
                'transaction_count': len(customer_data),
                'total_spent': customer_data['price'].sum(),
                'avg_order_value': customer_data['price'].mean(),
                'predicted_age': prediction['age_prediction'],
                'age_confidence': prediction['age_probabilities'][prediction['age_prediction']],
                'predicted_gender': prediction['gender_prediction'],
                'gender_confidence': prediction['gender_probabilities'][prediction['gender_prediction']],
                'overall_confidence': prediction['confidence']
            })

            customer_count += 1
            if customer_count % 1000 == 0:
                print(f"Processed {customer_count} customers...")

        results_df = pd.DataFrame(results)
        return results_df

    def generate_insights_report(self, results_df):

        print("\n=== DEMOGRAPHIC INSIGHTS REPORT ===")


        print("\nPredicted Age Distribution:")
        age_dist = results_df['predicted_age'].value_counts(normalize=True) * 100
        for age, pct in age_dist.items():
            print(f"  {age}: {pct:.1f}%")

        print("\nPredicted Gender Distribution:")
        gender_dist = results_df['predicted_gender'].value_counts(normalize=True) * 100
        for gender, pct in gender_dist.items():
            print(f"  {gender}: {pct:.1f}%")


        high_conf = results_df[results_df['overall_confidence'] > 0.7]
        print(
            f"\nHigh Confidence Predictions: {len(high_conf)} customers ({len(high_conf) / len(results_df) * 100:.1f}%)")


        print("\nSpending Patterns by Age Group:")
        age_spending = results_df.groupby('predicted_age')['avg_order_value'].mean()
        for age, avg_spend in age_spending.items():
            print(f"  {age}: ₹{avg_spend:.0f} average order value")

        print("\nSpending Patterns by Gender:")
        gender_spending = results_df.groupby('predicted_gender')['avg_order_value'].mean()
        for gender, avg_spend in gender_spending.items():
            print(f"  {gender}: ₹{avg_spend:.0f} average order value")


        print(f"\nSynergy Opportunities:")
        print(
            f"High-value young customers (<25, >₹5000 AOV): {len(results_df[(results_df['predicted_age'] == 'under_25') & (results_df['avg_order_value'] > 5000)])}")
        print(
            f"Frequent mature customers (40+, >3 transactions): {len(results_df[(results_df['predicted_age'] == 'over_40') & (results_df['transaction_count'] > 3)])}")

        return results_df



def demo_prediction_system(df):


    print("=== DEMOGRAPHIC PREDICTION SYSTEM DEMO ===")


    predictor = DemographicPredictor()


    sample_users = df['user'].unique()[:100]
    sample_df = df[df['user'].isin(sample_users)]


    results = predictor.analyze_customer_base(sample_df)


    final_results = predictor.generate_insights_report(results)


    print(f"\n=== SAMPLE INDIVIDUAL PREDICTIONS ===")
    for i in range(min(5, len(results))):
        row = results.iloc[i]
        print(f"\nCustomer {row['user_id']}:")
        print(f"  Transactions: {row['transaction_count']}")
        print(f"  Total Spent: ₹{row['total_spent']:.0f}")
        print(f"  Predicted: {row['predicted_age']}, {row['predicted_gender']}")
        print(f"  Confidence: {row['overall_confidence']:.2f}")

    return results




print("Demographics Prediction Framework loaded successfully!")
print("Usage: predictor = DemographicPredictor()")
print("       results = predictor.analyze_customer_base(your_dataframe)")


# Validation Framework Implementation
class ModelValidator:


    def __init__(self, predictor):
        self.predictor = predictor

    def validate_predictions(self, df, actual_demographics_df):


        predictions = self.predictor.analyze_customer_base(df)


        validation_data = predictions.merge(
            actual_demographics_df,
            left_on='user_id',
            right_on='user_id',
            how='inner'
        )


        metrics = self._calculate_metrics(validation_data)

        return metrics, validation_data

    def _calculate_metrics(self, validation_data):

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

        metrics = {}


        age_accuracy = accuracy_score(validation_data['actual_age'], validation_data['predicted_age'])
        age_precision, age_recall, age_f1, _ = precision_recall_fscore_support(
            validation_data['actual_age'], validation_data['predicted_age'], average='weighted', zero_division=0
        )


        gender_accuracy = accuracy_score(validation_data['actual_gender'], validation_data['predicted_gender'])
        gender_precision, gender_recall, gender_f1, _ = precision_recall_fscore_support(
            validation_data['actual_gender'], validation_data['predicted_gender'], average='weighted', zero_division=0
        )


        combined_correct = (
                (validation_data['actual_age'] == validation_data['predicted_age']) &
                (validation_data['actual_gender'] == validation_data['predicted_gender'])
        ).sum()
        combined_accuracy = combined_correct / len(validation_data)


        age_baseline = 1 / 3  # 3 age groups
        gender_baseline = 1 / 2  # 2 genders
        combined_baseline = age_baseline * gender_baseline

        metrics = {
            'age_accuracy': round(age_accuracy, 3),
            'age_precision': round(age_precision, 3),
            'age_recall': round(age_recall, 3),
            'age_f1': round(age_f1, 3),
            'age_lift': round(age_accuracy / age_baseline, 2),

            'gender_accuracy': round(gender_accuracy, 3),
            'gender_precision': round(gender_precision, 3),
            'gender_recall': round(gender_recall, 3),
            'gender_f1': round(gender_f1, 3),
            'gender_lift': round(gender_accuracy / gender_baseline, 2),

            'combined_accuracy': round(combined_accuracy, 3),
            'combined_lift': round(combined_accuracy / combined_baseline, 2),

            'sample_size': len(validation_data)
        }


        age_cm = confusion_matrix(validation_data['actual_age'], validation_data['predicted_age'])
        gender_cm = confusion_matrix(validation_data['actual_gender'], validation_data['predicted_gender'])

        metrics['age_confusion_matrix'] = age_cm
        metrics['gender_confusion_matrix'] = gender_cm

        return metrics

    def print_validation_report(self, metrics):

        print("\n=== MODEL VALIDATION REPORT ===")
        print(f"Sample Size: {metrics['sample_size']} customers")

        print(f"\nAGE GROUP PREDICTION PERFORMANCE:")
        print(f"  Accuracy: {metrics['age_accuracy']:.1%} (vs {33.3:.1f}% random baseline)")
        print(f"  Precision: {metrics['age_precision']:.3f}")
        print(f"  Recall: {metrics['age_recall']:.3f}")
        print(f"  F1-Score: {metrics['age_f1']:.3f}")
        print(f"  Lift over Random: {metrics['age_lift']:.1f}x")

        print(f"\nGENDER PREDICTION PERFORMANCE:")
        print(f"  Accuracy: {metrics['gender_accuracy']:.1%} (vs 50.0% random baseline)")
        print(f"  Precision: {metrics['gender_precision']:.3f}")
        print(f"  Recall: {metrics['gender_recall']:.3f}")
        print(f"  F1-Score: {metrics['gender_f1']:.3f}")
        print(f"  Lift over Random: {metrics['gender_lift']:.1f}x")

        print(f"\nCOMBINED PREDICTION PERFORMANCE:")
        print(f"  Accuracy: {metrics['combined_accuracy']:.1%} (vs {16.7:.1f}% random baseline)")
        print(f"  Lift over Random: {metrics['combined_lift']:.1f}x")

        print(f"\nAge Group Confusion Matrix:")
        print(metrics['age_confusion_matrix'])

        print(f"\nGender Confusion Matrix:")
        print(metrics['gender_confusion_matrix'])



def analyze_prediction_confidence(results_df):

    print("\n=== CONFIDENCE ANALYSIS ===")


    confidence_bins = pd.cut(results_df['overall_confidence'],
                             bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                             labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    conf_dist = confidence_bins.value_counts(normalize=True) * 100
    print("Confidence Distribution:")
    for conf_level, pct in conf_dist.items():
        print(f"  {conf_level}: {pct:.1f}%")


    high_conf = results_df[results_df['overall_confidence'] > 0.7]
    if len(high_conf) > 0:
        print(f"\nHigh Confidence Customers Characteristics:")
        print(f"  Average Transactions: {high_conf['transaction_count'].mean():.1f}")
        print(f"  Average Spending: ₹{high_conf['total_spent'].mean():.0f}")
        print(f"  Average Order Value: ₹{high_conf['avg_order_value'].mean():.0f}")

    return confidence_bins


def generate_business_recommendations(results_df):

    print("\n=== BUSINESS RECOMMENDATIONS ===")


    total_customers = len(results_df)
    total_revenue = results_df['total_spent'].sum()


    age_segments = results_df.groupby('predicted_age').agg({
        'user_id': 'count',
        'total_spent': ['sum', 'mean'],
        'avg_order_value': 'mean',
        'overall_confidence': 'mean'
    }).round(2)

    age_segments.columns = ['Customer_Count', 'Total_Revenue', 'Avg_LTV', 'Avg_AOV', 'Avg_Confidence']
    age_segments['Revenue_Share'] = (age_segments['Total_Revenue'] / total_revenue * 100).round(1)

    print("Age Segment Analysis:")
    print(age_segments)


    gender_segments = results_df.groupby('predicted_gender').agg({
        'user_id': 'count',
        'total_spent': ['sum', 'mean'],
        'avg_order_value': 'mean',
        'overall_confidence': 'mean'
    }).round(2)

    gender_segments.columns = ['Customer_Count', 'Total_Revenue', 'Avg_LTV', 'Avg_AOV', 'Avg_Confidence']
    gender_segments['Revenue_Share'] = (gender_segments['Total_Revenue'] / total_revenue * 100).round(1)

    print("\nGender Segment Analysis:")
    print(gender_segments)


    print(f"\nSTRATEGIC RECOMMENDATIONS:")


    high_value_age = age_segments.loc[age_segments['Avg_LTV'].idxmax()]
    high_value_gender = gender_segments.loc[gender_segments['Avg_LTV'].idxmax()]

    print(f"\n1. PRIORITY TARGETING:")
    print(f"   Focus on {high_value_age.name} age group (₹{high_value_age['Avg_LTV']:.0f} avg LTV)")
    print(f"   Focus on {high_value_gender.name} customers (₹{high_value_gender['Avg_LTV']:.0f} avg LTV)")


    low_conf_customers = len(results_df[results_df['overall_confidence'] < 0.5])
    print(f"\n2. DATA COLLECTION PRIORITY:")
    print(f"   {low_conf_customers} customers need better demographic signals")
    print(f"   Focus on customers with 1-2 transactions for engagement")


    print(f"\n3. SYNERGY OPPORTUNITIES:")
    young_big_spenders = len(results_df[
                                 (results_df['predicted_age'] == 'under_25') &
                                 (results_df['avg_order_value'] > results_df['avg_order_value'].quantile(0.75))
                                 ])

    mature_frequent = len(results_df[
                              (results_df['predicted_age'] == 'over_40') &
                              (results_df['transaction_count'] > 3)
                              ])

    print(f"   Target {young_big_spenders} young high-spenders for premium upselling")
    print(f"   Engage {mature_frequent} mature frequent buyers for loyalty programs")


def create_customer_personas(results_df):

    print("\n=== CUSTOMER PERSONAS ===")

    personas = []


    for age in results_df['predicted_age'].unique():
        for gender in results_df['predicted_gender'].unique():
            segment_data = results_df[
                (results_df['predicted_age'] == age) &
                (results_df['predicted_gender'] == gender)
                ]

            if len(segment_data) > 0:
                persona = {
                    'segment': f"{age}_{gender}",
                    'size': len(segment_data),
                    'avg_transactions': segment_data['transaction_count'].mean(),
                    'avg_spending': segment_data['total_spent'].mean(),
                    'avg_order_value': segment_data['avg_order_value'].mean(),
                    'confidence': segment_data['overall_confidence'].mean()
                }
                personas.append(persona)

    personas_df = pd.DataFrame(personas).round(2)
    personas_df = personas_df.sort_values('avg_spending', ascending=False)

    print("Customer Personas (ranked by spending):")
    print(personas_df)

    # Detailed persona descriptions
    for _, persona in personas_df.head(3).iterrows():
        age, gender = persona['segment'].split('_')
        print(f"\n{age.replace('_', '-').title()} {gender.title()} Persona:")
        print(f"  Size: {persona['size']} customers ({persona['size'] / len(results_df) * 100:.1f}%)")
        print(f"  Behavior: {persona['avg_transactions']:.1f} transactions, ₹{persona['avg_order_value']:.0f} AOV")
        print(f"  Value: ₹{persona['avg_spending']:.0f} total lifetime spend")
        print(f"  Prediction Confidence: {persona['confidence']:.2f}")


def complete_demo_system(df):


    print("=== COMPREHENSIVE DEMOGRAPHIC ANALYSIS SYSTEM ===")


    predictor = DemographicPredictor()


    print("\n1. Running demographic predictions...")
    results = predictor.analyze_customer_base(df)


    print("\n2. Generating demographic insights...")
    final_results = predictor.generate_insights_report(results)


    print("\n3. Analyzing prediction confidence...")
    confidence_analysis = analyze_prediction_confidence(results)


    print("\n4. Generating business recommendations...")
    generate_business_recommendations(results)


    print("\n5. Creating customer personas...")
    create_customer_personas(results)


    print(f"\n6. Sample Individual Predictions:")
    high_conf_samples = results[results['overall_confidence'] > 0.6].head(3)

    for _, row in high_conf_samples.iterrows():

        customer_data = df[df['user'] == row['user_id']]
        detailed_prediction = predictor.predict_demographics(customer_data)

        print(f"\nCustomer {row['user_id']}:")
        print(f"  Transactions: {row['transaction_count']}")
        print(f"  Total Spent: ₹{row['total_spent']:.0f}")
        print(
            f"  Age Prediction: {row['predicted_age']} ({detailed_prediction['age_probabilities'][row['predicted_age']]:.1%} confidence)")
        print(
            f"  Gender Prediction: {row['predicted_gender']} ({detailed_prediction['gender_probabilities'][row['predicted_gender']]:.1%} confidence)")
        print(f"  Overall Confidence: {row['overall_confidence']:.2f}")


        customer_categories = customer_data['generic_name'].value_counts().head(2)
        print(f"  Top Categories: {', '.join([f'{cat} ({count})' for cat, count in customer_categories.items()])}")

    return results



def create_synthetic_validation_data(results_df, accuracy_simulation=0.7):

    np.random.seed(42)

    synthetic_actual = []

    for _, row in results_df.iterrows():
        # Simulate actual age (with some accuracy)
        if np.random.random() < accuracy_simulation:
            actual_age = row['predicted_age']  # Correct prediction
        else:
            ages = ['under_25', '25_to_40', 'over_40']
            ages.remove(row['predicted_age'])
            actual_age = np.random.choice(ages)


        if np.random.random() < accuracy_simulation:
            actual_gender = row['predicted_gender']  # Correct prediction
        else:
            actual_gender = 'female' if row['predicted_gender'] == 'male' else 'male'

        synthetic_actual.append({
            'user_id': row['user_id'],
            'actual_age': actual_age,
            'actual_gender': actual_gender
        })

    return pd.DataFrame(synthetic_actual)



def demo_validation_system(df, results_df):

    print("\n=== VALIDATION SYSTEM DEMO ===")


    actual_demographics = create_synthetic_validation_data(results_df, accuracy_simulation=0.68)


    predictor = DemographicPredictor()
    validator = ModelValidator(predictor)


    validation_data = results_df.merge(actual_demographics, on='user_id', how='inner')
    metrics = validator._calculate_metrics(validation_data)


    validator.print_validation_report(metrics)

    return metrics, validation_data


print("\nComplete Demographics Prediction Framework loaded!")
print("\nUsage Examples:")
print("1. Basic Analysis: predictor = DemographicPredictor(); results = predictor.analyze_customer_base(df)")
print("2. Complete Demo: results = complete_demo_system(df)")
print("3. Validation Demo: metrics = demo_validation_system(df, results)")


__all__ = [
    'DemographicPredictor',
    'ModelValidator',
    'complete_demo_system',
    'demo_validation_system',
    'analyze_prediction_confidence',
    'generate_business_recommendations',
    'create_customer_personas'
]