import numpy as np
import pandas as pd
import streamlit as st

def ahp_weights(pairwise_matrix):
    col_sum = pairwise_matrix.sum(axis=0)
    normalized_matrix = pairwise_matrix / col_sum
    return normalized_matrix.mean(axis=1)

def consistency_ratio(pairwise_matrix, weights):
    n = pairwise_matrix.shape[0]
    RI_dict = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
    Aw = np.dot(pairwise_matrix, weights)
    lambda_max = np.mean(Aw / weights)
    CI = (lambda_max - n) / (n - 1)
    return (CI / RI_dict.get(n, 1.24)) if n in RI_dict else 0

def topsis_normalize(decision_matrix):
    return decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))

def ideal_solutions(weighted_matrix, criteria_types):
    ideal = []
    negative_ideal = []
    for j, ctype in enumerate(criteria_types):
        if ctype == 'benefit':
            ideal.append(weighted_matrix[:, j].max())
            negative_ideal.append(weighted_matrix[:, j].min())
        else:
            ideal.append(weighted_matrix[:, j].min())
            negative_ideal.append(weighted_matrix[:, j].max())
    return np.array(ideal), np.array(negative_ideal)

def main():
    st.title("AHP-TOPSIS Decision Assistant")
    
    # Step 1: Input Criteria and Alternatives
    with st.expander("1. Define Criteria & Alternatives"):
        criteria = st.text_input("Criteria (comma-separated):", "Cost,Quality,Sustainability")
        alternatives = st.text_input("Alternatives (comma-separated):", "Option1,Option2,Option3")
        criteria_list = [c.strip() for c in criteria.split(',')]
        alt_list = [a.strip() for a in alternatives.split(',')]

    # Step 2: AHP Pairwise Comparisons
    with st.expander("2. Criteria Comparisons (AHP)"):
        st.write("Compare criteria using Saaty's scale (1-9):")
        pairwise_matrix = np.ones((len(criteria_list), len(criteria_list)))
        for i in range(len(criteria_list)):
            for j in range(i+1, len(criteria_list)):
                val = st.slider(
                    f"{criteria_list[i]} vs {criteria_list[j]}", 
                    1/9, 9.0, 1.0, 0.1,
                    format="%0.1f"
                )
                pairwise_matrix[i,j] = val
                pairwise_matrix[j,i] = 1/val

    # Step 3: Decision Matrix Input
    with st.expander("3. Alternative Ratings"):
        st.write("Enter performance values for each alternative:")
        decision_matrix = []
        for alt in alt_list:
            row = []
            for crit in criteria_list:
                val = st.number_input(f"{alt} - {crit}", value=1.0)
                row.append(val)
            decision_matrix.append(row)
        decision_matrix = np.array(decision_matrix)

    # Step 4: Criteria Type Selection
    with st.expander("4. Specify Criteria Types"):
        criteria_types = [st.selectbox(f"{crit} type:", ['cost', 'benefit']) 
                         for crit in criteria_list]

    if st.button("Calculate Rankings"):
        # AHP Calculations
        weights = ahp_weights(pairwise_matrix)
        cr = consistency_ratio(pairwise_matrix, weights)
        
        # TOPSIS Calculations
        norm_matrix = topsis_normalize(decision_matrix)
        weighted_matrix = norm_matrix * weights
        ideal, neg_ideal = ideal_solutions(weighted_matrix, criteria_types)
        dist_p = np.sqrt(((weighted_matrix - ideal)**2).sum(axis=1))
        dist_n = np.sqrt(((weighted_matrix - neg_ideal)**2).sum(axis=1))
        closeness = dist_n / (dist_p + dist_n)
        closeness = pd.Series(closeness)
        
        # Display Results
        st.subheader("Results")
        df = pd.DataFrame({
            'Alternative': alt_list,
            'Score': closeness,
            'Rank': closeness.rank(ascending=False).astype(int)
        }).sort_values('Rank')
        st.dataframe(df.style.background_gradient(cmap='Blues', subset=['Score']))
        
        st.write(f"**Criteria Weights:** {np.round(weights, 3)}")
        st.write(f"**Consistency Ratio:** {cr:.3f} {'✅' if cr <= 0.1 else '⚠️ Revise comparisons!'}")

if __name__ == "__main__":
    main()


