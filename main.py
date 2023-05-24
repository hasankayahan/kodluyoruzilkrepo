import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


st.set_page_config(page_title="Hitters Salary Prediction", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "EDA & Visualizations", "Contact"])

if page == "Prediction":
    st.title("Hotel Reservations")

    # Modeli yükleyin

    from joblib import load

    # Modelin tam yolunu kullanarak modeli yükleyin
    model = lgb.Booster(model_file='C:\\Users\\hasan\\PycharmProjects\\introductionToDataScience\\lgbm_model.txt')



    # Kullanıcıdan girdi alın
    # Özellikleri belirtin ve değerlerini isteyin
    # Örnek olarak 'AtBat', 'Hits' ve 'HmRun' özelliklerini kullanalım

    def user_input_features():
        no_of_adults = st.sidebar.number_input("no_of_adults", min_value=1, step=1)
        no_of_children = st.sidebar.number_input("no_of_children", min_value=0, step=1)
        no_of_weekend_nights = st.sidebar.number_input("no_of_weekend_nights", min_value=0, step=1)
        no_of_week_nights = st.sidebar.number_input("no_of_week_nights", min_value=0, step=1)
        type_of_meal_plan = st.sidebar.selectbox("type_of_meal_plan", ('Not Selected', 'Meal Plan 1', "Meal Plan 2", "Meal Plan 3"))
        required_car_parking_space = st.sidebar.selectbox('required_car_parking_space', ('0', '1'))
        room_type_reserved = st.sidebar.selectbox('room_type_reserved', ('Room_Type 1', 'Room_Type 2', "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"))
        lead_time = st.sidebar.number_input('lead_time', min_value=1, step=1)
        arrival_year = st.sidebar.selectbox('arrival_year', ("2017", "2018"))
        arrival_month = st.sidebar.selectbox('arrival_month', ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"))
        arrival_date = st.slider("arrival_date", 1, 31, 1)
        market_segment_type = st.sidebar.selectbox('market_segment_type', ('Online', 'Offline', "Corporate", "Complementary", "Aviation"))
        repeated_guest = st.sidebar.selectbox('repeated_guest', ("0", "1"))
        no_of_previous_cancellations = st.sidebar.number_input("no_of_previous_cancellations", min_value=0, step=1)
        no_of_previous_bookings_not_canceled = st.sidebar.number_input("no_of_previous_bookings_not_canceled", min_value=0, step=1)
        avg_price_per_room = st.sidebar.slider("avg_price_per_room", 60.0, 380.93, 500.0)
        no_of_special_requests = st.sidebar.selectbox('no_of_special_requests', ("1", "2", "3", "4", "5"))
        data = {"no_of_adults": no_of_adults,
                'no_of_children': no_of_children,
                'no_of_weekend_nights': no_of_weekend_nights,
                'no_of_week_nights': no_of_week_nights,
                'type_of_meal_plan': type_of_meal_plan,
                'required_car_parking_space': required_car_parking_space,
                'room_type_reserved': room_type_reserved,
                'lead_time': lead_time,
                'arrival_year': arrival_year,
                'arrival_month': arrival_month,
                'arrival_date': arrival_date,
                'market_segment_type': market_segment_type,
                'repeated_guest': repeated_guest,
                'no_of_previous_cancellations': no_of_previous_cancellations,
                'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
                'avg_price_per_room': avg_price_per_room,
                'no_of_special_requests': no_of_special_requests}

        features = pd.DataFrame(data, index=[0])
        return features


    input_df = user_input_features()

    st.dataframe(input_df)

    # Veri setini yükle
    url = "https://raw.githubusercontent.com/Cat4VP/Hotel-Reservations-Dataset/main/Hotel%20Reservations.csv"
    df = pd.read_csv(url)

    birlesmis = pd.concat([input_df, df], axis=0)
    birlesmis.drop("Booking_ID", inplace=True, axis=1)
    birlesmis.reset_index(drop=True, inplace=True)
    birlesmis.index += 1
    birlesmis.reset_index()
    st.write("Kullanıcı bilgilerini ve veri setini birleştirelim.")
    st.write(birlesmis.head())


    def outlier_thresholds(df, col_name, q1=0.25, q3=0.75):
        quartile1 = df[col_name].quantile(q1)
        quartile3 = df[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit


    def replace_with_thresholds(df, variable):
        low_limit, up_limit = outlier_thresholds(df, variable)
        df.loc[(df[variable] < low_limit), variable] = low_limit
        df.loc[(df[variable] > up_limit), variable] = up_limit


    def check_outlier(df, col_name):
        low_limit, up_limit = outlier_thresholds(df, col_name)
        if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False


    def grab_col_names(df, cat_th=10, car_th=20):
        """

        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            df: df
                    Değişken isimleri alınmak istenilen df
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

        """

        # cat_cols, cat_but_car
        cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
        num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
                       df[col].dtypes != "O"]
        cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and
                       df[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in df.columns if df[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {df.shape[0]}")
        print(f"Variables: {df.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')
        return cat_cols, num_cols, cat_but_car


    cat_cols, num_cols, cat_but_car = grab_col_names(birlesmis)

    #############################################
    # 3. Feature Extraction (Özellik Çıkarımı)
    #############################################

    ## toplam ksii sayisi
    birlesmis['NEW_no_of_total_person'] = birlesmis["no_of_children"] + birlesmis["no_of_adults"]
    ## toplam kalinan gun sayisi
    birlesmis['NEW_total_number_of_stayed_days'] = birlesmis["no_of_weekend_nights"] + birlesmis["no_of_week_nights"]
    ## toplam kalinan gun sayisi
    birlesmis['NEW_total_number_of_stayed_days'] = birlesmis["no_of_weekend_nights"] + birlesmis["no_of_week_nights"]
    ## kac hafta kalmis kac gun kalmis
    birlesmis['NEW_number_weeks'], birlesmis['NEW_number_days'] = divmod(birlesmis['NEW_total_number_of_stayed_days'], 7)
    ### season
    birlesmis["arrival_month"] = birlesmis["arrival_month"].astype("int64")
    birlesmis["NEW_season"] = ["Winter" if col <= 2 else "Spring" if 3 <= col <= 5 else "Summer" if 6 <= col <= 8 else "Fall" if 9 <= col <= 11 else "Winter" for col in birlesmis["arrival_month"]]
    ## yemek planı var mı yok mu ?
    birlesmis["NEW_meal_plan"] = [0 if col == 'Not Selected' else 1 for col in birlesmis["type_of_meal_plan"]]
    ##  oda segmenti
    birlesmis["NEW_Room_Type"] = pd.qcut(birlesmis['avg_price_per_room'].rank(method="first"), 3, labels=['Economic_Room', 'Standard_Room', 'Luxury_Rooom'])
    ## Churn degerini sayisallastir.
    birlesmis.booking_status = df.booking_status.replace({"Not_Canceled": 0, "Canceled": 1})
    ### gelme durumlarina gore derecelnediyor
    #grby_month_book = birlesmis.groupby(["arrival_month", "booking_status"]).count().reset_index()
    #list_of_months = []
    #for i in range(1, 13):
    #    x = grby_month_book[grby_month_book.arrival_month == i]
    #    rate = x[x.booking_status == 1].Booking_ID.values / x[x.booking_status == 0].Booking_ID.values * 100
    #    list_of_months.append(rate)
    #pd.Series(list_of_months)
    birlesmis["NEW_arrival_month_rate"] = birlesmis.arrival_month.replace(
        {3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 9: 1, 1: 0, 2: 0, 8: 0, 10: 0, 11: 0, 12: 0})

    # Misafir başına düşen ortalama fiyatı hesaplama:
    birlesmis['NEW_average_price_per_guest'] = birlesmis['avg_price_per_room'] / birlesmis['NEW_no_of_total_person']

    # Booking Time Category (booking_time_category) sütunu oluşturma
    birlesmis['booking_time_category'] = pd.cut(birlesmis['lead_time'], bins=[-float('inf'), 30, 60, float('inf')], labels=['Erken', 'Orta', 'Gec'])

    ##  temel bilesen analizi degerleri

    pca = PCA(n_components=1)
    NEW_pca_no_people = pca.fit_transform(birlesmis[["no_of_adults", "no_of_children"]])
    birlesmis["NEW_pca_no_people"] = NEW_pca_no_people

    NEW_pca_no_week = pca.fit_transform(birlesmis[["no_of_weekend_nights", "no_of_week_nights"]])
    birlesmis["NEW_no_of_week_days"] = NEW_pca_no_week

    # özel istek var mı yok mu
    birlesmis["no_of_special_requests"] = birlesmis["no_of_special_requests"].astype("int64")
    birlesmis["NEW_flag_special_requests"] = [1 if col > 0 else 0 for col in birlesmis["no_of_special_requests"]]

    birlesmis['NEW_Total_Price'] = (birlesmis["no_of_weekend_nights"] + birlesmis["no_of_week_nights"]) * birlesmis["avg_price_per_room"]
    birlesmis["NEW_Total_Price_Per"] = birlesmis["NEW_Total_Price"] / birlesmis["NEW_no_of_total_person"]

    cat_cols, num_cols, cat_but_car = grab_col_names(birlesmis)




    # Robust Scaler
    cols = [col for col in birlesmis.columns if col not in ["booking_status",
                                                     "type_of_meal_plan",
                                                     "room_type_reserved",
                                                     "market_segment_type",
                                                     "NEW_season",
                                                     "NEW_Room_Type",
                                                     "booking_time_category"]]

    for col in cols:
        birlesmis[col] = birlesmis[col].astype("float64")

    for col in cols:
        birlesmis[col] = RobustScaler().fit_transform(birlesmis[[col]])



    encoder = ["type_of_meal_plan","room_type_reserved","market_segment_type","NEW_season","NEW_Room_Type", "booking_time_category"]

    def one_hot_encoding(dataframe, categorical_cols, drop_first=True):
        dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
        return dataframe

    birlesmis = one_hot_encoding(birlesmis, encoder)

    cat_cols, num_cols, cat_but_car = grab_col_names(birlesmis)


    # inputa fe uyguladıktan sonra çekelim predict için

    tahmin_edilecek = birlesmis[:1]
    # model_icin = birlesmis[1:]

    st.write("------------------------------")



    prediction = model.predict(tahmin_edilecek, predict_disable_shape_check=True)


    #5 basamaklı
    #st.title('Otel Rezervasyon Churn Tahmini :hotel:')
    #tahminler_df = pd.DataFrame({'Tahmin': prediction})
    #st.write(tahminler_df)

    # 2 basamaklı
    st.title('Hotel Booking Churn Forecast :hotel:')
    tahminler_df = pd.DataFrame({'Tahmin': prediction})
    tahminler_df['Tahmin'] = tahminler_df['Tahmin'].apply(lambda x: round(x, 2))
    st.write(tahminler_df)



elif page == "EDA & Visualizations":
    st.title("EDA & Visualizations")

    # Add your EDA and visualizations here

    # Veri kümesini yükleyin
    url = "https://raw.githubusercontent.com/Cat4VP/Hotel-Reservations-Dataset/main/Hotel%20Reservations.csv"
    data = pd.read_csv(url)


    # Streamlit başlığı
    st.title("Hotel Reservations Dataset Statistical Reports and Visualizations")

    # Veri kümesini göster
    st.write("### Dataset:")
    st.write(data.head())

    show_profile_report = st.checkbox("Pandas Profiling Report Show")

    if show_profile_report:
        st.write("### Pandas Profiling Report Show:")
        profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
        st_profile_report(profile)

        # Veri kümesinin özellik dağılımlarını göster
        st.write("### Özellik Dağılımları:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=data, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Sütunlar arasındaki ilişkileri göster
        st.write("### Sütunlar Arası İlişkiler:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.pairplot(data=data, diag_kind="kde")
        st.pyplot(fig)

        # Korelasyon matrisini göster
        st.write("### Korelasyon Matrisi:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
        st.pyplot(fig)






else:
    st.title("Contact")

    col1, col2, = st.columns(2)

    with col1:
        st.title("Miuul")
        image_url = "https://is1-ssl.mzstatic.com/image/thumb/Podcasts116/v4/af/54/f1/af54f165-8122-2804-f685-1e46b269a080/mza_6214609086229939736.jpg/313x0w.webp"
        st.image(image_url, use_column_width=True)

    with col2:
        st.title("Veri Bilimi Okulu")
        image_url = "https://www.miuul.com/image/store/xl_motivasyon-kupasi-6452274e6ae9b.png"
        st.image(image_url, use_column_width=True)

    col1, col2 = st.columns(2)

    with col1:
        link = "[Miuul](https://miuul.com)"
        st.markdown(link, unsafe_allow_html=True)

    with col2:
        link = "[Veri Bilimi Okulu](https://bootcamp.veribilimiokulu.com/bootcamp-programlari/veri-bilimci-yetistirme-programi/)"
        st.markdown(link, unsafe_allow_html=True)

    col1, = st.columns(1)

    with col1:
        st.title("Kaynakça 🎤")
        video_url = "https://www.youtube.com/watch?v=Ww9M0WJfGN8"
        st.video(video_url)

    col3, = st.columns(1)
    with col3:
        st.title("Linkedin")
        st.write("[Linkedin](https://www.linkedin.com/in/hasankayahan/)")

    # Data Vaders :)

    st.title("Best Group :) DATA VADERS")

    video_path = "C:\\Users\\hasan\\OneDrive\\Desktop\\Başvuru.mp4"

    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    import streamlit as st
    import base64

    st.title("Contact")

    # ... Diğer içerikler ...

    # PDF dosyasını ekleyin
    pdf_path = "C:\\Users\\hasan\\OneDrive\\Desktop\\Siyah Beyaz Modern Teknoloji Şirketi Logosu-1.pdf"
    pdf_file = open(pdf_path, 'rb')
    pdf_bytes = pdf_file.read()

    # Seçenek düğmesiyle PDF'nin indirme bağlantısını oluşturun
    show_pdf = st.checkbox("PDF Dosyasını Göster")

    if show_pdf:
        st.markdown("### PDF Dosyası:")
        b64_pdf = base64.b64encode(pdf_bytes).decode('latin-1')
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Siyah Beyaz Modern Teknoloji Şirketi Logosu-1.pdf">PDF İndir</a>'
        st.markdown(href, unsafe_allow_html=True)


if __name__ == '__user_input_features__':
    user_input_features()








