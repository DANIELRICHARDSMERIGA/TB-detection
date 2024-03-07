import streamlit as st

def app():

    st.title("TB Patient Remedies based on Air Quality Index")
    st.markdown("Enter the Air Quality Index value:")

    # Slider for AQI input
    aqi = st.slider("AQI", min_value=0, max_value=500, value=100, step=1)

    if aqi <= 50:
        st.success("The air quality is good. Patients with TB can follow their regular treatment plan.")
    elif aqi <= 100:
        st.info("The air quality is moderate. Patients with TB should limit outdoor activities.")
    elif aqi <= 150:
        st.warning("The air quality is unhealthy for sensitive groups. TB patients should avoid outdoor activities.")
    elif aqi <= 200:
        st.warning("The air quality is unhealthy. TB patients should stay indoors as much as possible.")
    else:
        st.error("The air quality is hazardous. TB patients should remain indoors, use air purifiers, and wear masks if going outside.")

    # Real time AQI measure
    st.sidebar.markdown(
    f'<a href="https://aqual.netlify.app/" target="_blank" style="display: inline-block; padding: 12px 20px; background-color: #4CAF50; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">Real-Time AQI Measure</a>',
    unsafe_allow_html=True
    )

    

    

 
