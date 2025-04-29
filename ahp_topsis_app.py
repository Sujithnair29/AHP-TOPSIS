import numpy as np
import pandas as pd
import streamlit as st
from fractions import Fraction
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
        st.markdown("#### ✍️ Input your decision factors")

        # Get theme to adapt colors
        theme_base = st.get_option("theme.base")
        dark_mode = theme_base == "dark"
        input_bg = "#262730" if dark_mode else "#f7f9fa"
        text_color = "#ffffff" if dark_mode else "#000000"
        border_color = "#4a4a4a" if dark_mode else "#cccccc"

        # Define custom CSS for input fields
        st.markdown(f"""
            <style>
            .custom-input {{
                background-color: {input_bg};
                color: {text_color};
                padding: 10px 15px;
                border: 1px solid {border_color};
                border-radius: 10px;
                font-size: 16px;
                width: 100%;
                box-sizing: border-box;
                margin-bottom: 15px;
            }}
            label {{
                font-weight: 600;
                font-size: 15px;
            }}
            </style>
        """, unsafe_allow_html=True)

        # Render text inputs using HTML for more control
        criteria = st.text_input("Criteria (comma-separated):", "Cost,Quality,Sustainability", key="criteria_input")
        alternatives = st.text_input("Alternatives (comma-separated):", "Option1,Option2,Option3", key="alternatives_input")

        criteria_list = [c.strip() for c in criteria.split(',')]
        alt_list = [a.strip() for a in alternatives.split(',')]

        st.markdown("<hr style='margin-top:10px; margin-bottom:10px;'>", unsafe_allow_html=True)
        st.markdown(f"🔹 <b>{len(criteria_list)}</b> criteria detected: <i>{', '.join(criteria_list)}</i>", unsafe_allow_html=True)
        st.markdown(f"🔸 <b>{len(alt_list)}</b> alternatives detected: <i>{', '.join(alt_list)}</i>", unsafe_allow_html=True)


    # Step 1: Input Criteria and Alternatives
    # with st.expander("1. Define Criteria & Alternatives"):
    #     criteria = st.text_input("Criteria (comma-separated):", "Cost,Quality,Sustainability")
    #     alternatives = st.text_input("Alternatives (comma-separated):", "Option1,Option2,Option3")
    #     criteria_list = [c.strip() for c in criteria.split(',')]
    #     alt_list = [a.strip() for a in alternatives.split(',')]

    # Step 2: AHP Pairwise Comparisons
    # Define discrete Saaty scale values including reciprocals
    saaty_values = [
        1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2,
        1.0,
        2, 3, 4, 5, 6, 7, 8, 9
    ]

    # Define corresponding textual descriptions
    saaty_phrases = {
    1/9: "is extremely less important than",
    1/8: "is very, very strongly less important than",
    1/7: "is very strongly less important than",
    1/6: "is strongly plus less important than",
    1/5: "is strongly less important than",
    1/4: "is moderately plus less important than",
    1/3: "is moderately less important than",
    1/2: "is slightly less important than",
    1.0: "is equally important as",
    2: "is slightly more important than",
    3: "is moderately more important than",
    4: "is moderately plus more important than",
    5: "is strongly more important than",
    6: "is strongly plus more important than",
    7: "is very strongly more important than",
    8: "is very, very strongly more important than",
    9: "is extremely more important than"
    }

    
    # Your criteria list
    def format_slider_label(x):
        return str(Fraction(x).limit_denominator()) if x < 1 else str(int(x))
    
    theme_base = st.get_option("theme.base")
    dark_mode = theme_base == "dark"

    # Set theme-aware styles
    bg_color = "rgba(255, 255, 255, 0.08)" if dark_mode else "#f0f2f6"
    text_color = "#ffffff" if dark_mode else "#000000"
    border_color = "rgba(255,255,255,0.2)" if dark_mode else "#cccccc"

    with st.expander("2. Criteria Comparisons (AHP)"):
        st.markdown("#### Relative importance of criteria using Saaty's scale (1–9 and reciprocals):")
        st.markdown("<div style='margin-bottom: 15px;'>Use the sliders below to indicate how much more important one criterion is over another.</div>", unsafe_allow_html=True)

        pairwise_matrix = np.ones((len(criteria_list), len(criteria_list)))

        for i in range(len(criteria_list)):
            for j in range(i + 1, len(criteria_list)):
                with st.container():
                    st.markdown(f"**{criteria_list[i]} vs {criteria_list[j]}**")
                    
                    col1, col2 = st.columns([4, 3])
                    with col1:
                        val = st.select_slider(
                            label=f"slider-{criteria_list[i]}-vs-{criteria_list[j]}",
                            options=saaty_values,
                            value=1.0,
                            format_func=format_slider_label,
                            label_visibility="collapsed"
                        )
                    with col2:
                        phrase = saaty_phrases.get(val, "")
                        st.markdown(
                        f"""
                        <div style='
                            padding: 10px;
                            border-radius: 6px;
                            background-color: {bg_color};
                            color: {text_color};
                            border: 1px solid {border_color};
                            font-weight: 500;
                        '>
                            <b>{criteria_list[i]}</b> {phrase} <b>{criteria_list[j]}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                         )
                        # st.markdown(
                        #     f"<div style='padding:8px; background-color:#f0f2f6; border-radius:6px;'><b>{criteria_list[i]}</b> {phrase} <b>{criteria_list[j]}</b></div>",
                        #     unsafe_allow_html=True
                        # )

                    st.markdown("<hr style='margin-top:10px; margin-bottom:20px;'>", unsafe_allow_html=True)

                    pairwise_matrix[i, j] = val
                    pairwise_matrix[j, i] = 1 / val

    # with st.expander("2. Criteria Comparisons (AHP)"):
    #     st.write("Compare criteria using Saaty's scale (1–9 and reciprocals):")
    #     pairwise_matrix = np.ones((len(criteria_list), len(criteria_list)))
        
    #     for i in range(len(criteria_list)):
    #         for j in range(i + 1, len(criteria_list)):
    #             col1, col2 = st.columns([3, 3])
    #             with col1:
    #                 val = st.select_slider(
    #                     f"{criteria_list[i]} vs {criteria_list[j]}",
    #                     options=saaty_values,
    #                     format_func=lambda x: f"{x:.3f}" if x < 1 else str(int(x)),
    #                     value=1.0
    #                 )
    #             with col2:
    #                 phrase = saaty_phrases.get(val, "")
    #                 st.markdown(f"**{criteria_list[i]} {phrase} {criteria_list[j]}**")
                
    #             pairwise_matrix[i, j] = val
    #             pairwise_matrix[j, i] = 1 / val  # Automatic reciprocal filling
    # Step 2: AHP Pairwise Comparisons

    # with st.expander("2. Criteria Comparisons (AHP)"):
    #     st.write("Compare criteria using Saaty's scale (1-9):")
    #     pairwise_matrix = np.ones((len(criteria_list), len(criteria_list)))
    #     for i in range(len(criteria_list)):
    #         for j in range(i+1, len(criteria_list)):
    #             val = st.slider(
    #                 f"{criteria_list[i]} vs {criteria_list[j]}", 
    #                 1/9, 9.0, 1.0, 0.1,
    #                 format="%0.1f"
    #             )
    #             pairwise_matrix[i,j] = val
    #             pairwise_matrix[j,i] = 1/val

    # Step 3: Decision Matrix Input
    with st.expander("3. Alternative Ratings"):
        st.markdown("#### Performance values of alternatives against criteria")
        st.markdown("Provide performance values for each alternative under every criterion:")

        # Theme-aware styling
        theme_base = st.get_option("theme.base")
        dark_mode = theme_base == "dark"
        card_bg = "#1e1e1e" if dark_mode else "#f8f9fa"
        text_color = "#ffffff" if dark_mode else "#000000"
        border_color = "#444444" if dark_mode else "#dddddd"

        decision_matrix = []

        for alt in alt_list:
            st.markdown(
                f"""
                <div style="background-color: {card_bg}; color: {text_color}; 
                            padding: 5px 10px; margin: 5px 0; border-radius: 6px; 
                            border: 1px solid {border_color};">
                    <h5 style="margin-bottom: 5px; font-size: 20px;">🔹 {alt}</h6>
                """, unsafe_allow_html=True
            )

            cols = st.columns(len(criteria_list))
            row = []
            for idx, crit in enumerate(criteria_list):
                with cols[idx]:
                    val = st.number_input(
                        f"{alt} - {crit}",
                        value=1.0,
                        step=0.1,
                        min_value=0.0,
                        key=f"{alt}_{crit}"
                    )
                    row.append(val)

            st.markdown("</div>", unsafe_allow_html=True)
            decision_matrix.append(row)

        decision_matrix = np.array(decision_matrix)

    # with st.expander("3. Alternative Ratings"):
    #     st.write("Enter performance values for each alternative:")
    #     decision_matrix = []
    #     for alt in alt_list:
    #         row = []
    #         for crit in criteria_list:
    #             val = st.number_input(f"{alt} - {crit}", value=1.0)
    #             row.append(val)
    #         decision_matrix.append(row)
    #     decision_matrix = np.array(decision_matrix)

    # Step 4: Criteria Type Selection
    with st.expander("4. Specify Criteria Types"):
        st.markdown("#### Define whether each criterion should be minimized or maximized")

        criteria_types = []

        for crit in criteria_list:
            col1, col2 = st.columns([2, 4])  # Wider column for dropdown
            with col1:
                st.markdown(
                    f"<div style='padding: 0.5rem 0; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;' title='{crit}'>{crit}</div>",
                    unsafe_allow_html=True
                )
            with col2:
                criteria_types.append(
                    st.selectbox(" ", ["Minimize", "Maximize"], key=f"{crit}_type")
                )
        


    # with st.expander("4. Specify Criteria Types"):
    #     criteria_types = [st.selectbox(f"{crit} type:", ['Minimize', 'Maximize']) 
    #                      for crit in criteria_list]

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


