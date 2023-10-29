import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def data_cleaning(df):
    #The problems the original data had
    st.write('### This data had:') 
    st.write('-',df.isna().sum().sum(),' missing values')
    if df.isna().sum().sum():
        df.dropna(inplace=True)
        st.write('Lines with missing values were removed')

    duplicated = df[df.duplicated()]
    st.write('-',len(duplicated),' duplicated lines')
    if len(duplicated):
        df.drop_duplicates(inplace=True)
        st.write('Duplicated liness were removed')

    return df

def visualization(df):
    # Loop pelas colunas do DataFrame

        colunas = df.columns

        colunas_x = []
        colunas_y = []

        for col1 in colunas:
            if df[col1].dtype == 'float64':
                    colunas_x.append(col1)
            else:
                colunas_y.append(col1)


        st.write('### Correlation of continuous variables')

        custom_color_scale = ["#FFD700", "#FFA500", "#FF6347", "#FF4500", "#8B0000"]

        correlacao = df[colunas_x].corr()
        fig = px.imshow(correlacao,color_continuous_scale='Greys')
        fig.update_layout(
            xaxis=dict(tickfont=dict(color='black', size=14)),
            yaxis=dict(tickfont=dict(color='black', size=14)),
            coloraxis_colorbar=dict(title=dict(font=dict(color='black', size=16))),
            coloraxis_colorbar_ticks="outside"
        )
        for i in range(len(correlacao.index)):
            for j in range(len(correlacao.columns)):
                if round(correlacao.iloc[i, j],2) > 0:
                    fig.add_annotation(
                        text=str(round(correlacao.iloc[i, j],2)),
                        x=j,
                        y=i,
                        showarrow=False,
                        font=dict(color='white', size=12)  # Ajuste a cor e tamanho da fonte conforme necessário
                    )
                else:
                    fig.add_annotation(
                        text=str(round(correlacao.iloc[i, j],2)),
                        x=j,
                        y=i,
                        showarrow=False,
                        font=dict(color='black', size=12)  # Ajuste a cor e tamanho da fonte conforme necessário
                    )
        st.write(fig)
                    
        st.write('### Scatterplot of continuous variables')

        coluna_x = st.selectbox("Selecione a coluna X:", colunas_x)
        coluna_y = st.selectbox("Selecione a coluna Y:", colunas_x)

        
        fig = px.scatter(df,x=coluna_x,y=coluna_y,color_discrete_sequence=['black'])
        fig.update_layout(
            xaxis=dict(tickfont=dict(color='black', size=14),
                       title=dict(font=dict(size=16, color='black',family="Arial, sans-serif"))),
            yaxis=dict(tickfont=dict(color='black', size=14),
                       title=dict(font=dict(size=16, color='black',family="Arial, sans-serif"))),
            
            coloraxis_colorbar=dict(title=dict(font=dict(color='black', size=16))),
            coloraxis_colorbar_ticks="outside"
        )
        st.write(fig)

        st.write('### Distribution of variables')
        coluna_x = st.selectbox("Selecione a coluna X:", df.columns)
        fig = px.histogram(df, x=coluna_x,color_discrete_sequence=['black'])
        fig.update_layout(
            xaxis=dict(tickfont=dict(color='black', size=14),
                       title=dict(font=dict(size=16, color='black',family="Arial, sans-serif"))),
            yaxis=dict(tickfont=dict(color='black', size=14),
                       title=dict(font=dict(size=16, color='black',family="Arial, sans-serif"))),
            coloraxis_colorbar=dict(title=dict(font=dict(color='black', size=16))),
            coloraxis_colorbar_ticks="outside"
        )
        fig.update_traces(marker_line_color='white', marker_line_width=1)
        st.write(fig)

def download_clean(df):
     if st.button("Download Cleaned Data as CSV"):
            # Crie um link para download
            cleaned_data = df.to_csv(index=False).encode()
            st.download_button(
                label="Click here to download",
                data=cleaned_data,
                file_name="cleaned_data.csv",
                key="download_button",
            )
     
def machine_learning(df):
     # Adicione uma opção de seleção para aplicar ou não o machine learning
    aplicar_ml = st.radio("Want to apply Machine Learning over this dataset?", ('Yes', 'No'),index=1)

    if aplicar_ml == 'Yes':
        # O código para aplicar machine learning iria aqui
        # Você pode adicionar componentes interativos para ajustar parâmetros, selecionar algoritmos, etc.
        st.write("Welcome to the ML world")

        target=st.selectbox("Which column is your target?", df.columns)
        # Exemplo: Model Training
        X = df.drop(target,axis=1)
        y = df[target]
        problema = st.selectbox("Which ML problem is this?", ['Classificação','Regressão'])

        if problema == 'Classificação':
            modelo_selecionado = st.selectbox("Which ML model do you want to use?", ['DecisionTreeClassifier','LogisticRegression'])
            if modelo_selecionado == 'DecisionTreeClassifier':
                model = DecisionTreeClassifier()
            else:
                model = LogisticRegression()
             
        else:
            modelo_selecionado = st.selectbox("Which ML model do you want to use?", ['LinearRegression','DecisionTreeRegressor'])
            if modelo_selecionado == 'LinearRegression':
                model = LinearRegression()
                    
            else:
                model = DecisionTreeRegressor()
                    

        if st.button("Apply simple ML"):
            st.write(problema)
            X_treino, X_teste, y_treino, y_teste = train_test_split(X,y,test_size=0.3)
            model.fit(X_treino,y_treino)
            if problema == 'Classificação':
                st.write('Score: ',model.score(X_teste,y_teste))
            else:
                y_pred = model.predict(X_teste)
                st.write('R2-Score: ',r2_score(y_teste, y_pred))

    else:
        # Caso o usuário escolha "Não", você pode exibir o DataFrame original
        st.write("Go to hell")
        # Exibir o DataFrame original
        # st.write(df)

def main():
    st.title('Curious Unveil - a data exploratory tool')
    uploaded_file = st.file_uploader('Choose a tabular data csv file')


    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        st.write('### This are the first lines of the original dataset')
        st.write(df.head())

        df = data_cleaning(df)
        download_clean(df)

        st.write('### This are the main statistics of the data:')
        st.write(df.describe())

        visualization(df)

        machine_learning(df)


        

if __name__ == "__main__":
    main()