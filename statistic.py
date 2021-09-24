import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.linear_model import LinearRegression

#####クラスの定義#####

######関数の定義######
def choose_two_vari(data):
  vari_left, vari_right=st.beta_columns(2)
  vari1=vari_left.selectbox('Xの値',(data.columns.values))
  vari2=vari_right.selectbox('Yの値',(data.columns.values))

  return vari1, vari2

def choose_some_vari(data):
  vari_some=st.multiselect(
    '表示変数の選択',
    (data.columns.values))

  return vari_some

####################

#####画面表示#########

#タイトルを表示
st.title('Data Science')

#サイドバーを加える
# ファイルアップロード

uploaded_file = st.sidebar.file_uploader("ファイルをアップロード", type=['csv'])

if uploaded_file is not  None:

  add_selectbox = st.sidebar.selectbox(
    'Information',
    ('アップロードデータ','基本統計量','ヒストグラム','折れ線グラフ・棒グラフ','2変数の関係','群間比較','回帰分析','クラスタリング','機械学習')
  )
  df = pd.read_csv(uploaded_file)


  #各統計量の定義
  #平均
  mean_df=df.mean()
  #標準偏差
  std_df=df.std()
  #最大値
  max_df=df.max()
  #最小値
  min_df=df.min()

  #データインポートページ
  if add_selectbox=='アップロードデータ':
    #データのインポート
    st.header('アップロードデータ')

    #インポートしたデータの表示
    st.table(df)

  #基本統計量のページ
  if add_selectbox=='基本統計量':
    st.header('基本統計量')

    #基本統計量の日表示
    st.subheader('●平均、標準偏差、最大値、最小値')

    #データ要約
    data_summary=pd.DataFrame({
    '平均':mean_df,
    '標準偏差':std_df,
    '最大値':max_df,
    '最小値':min_df})

    #表示変数の選択
    desc_vari=choose_some_vari(df)
    st.table(data_summary.loc[desc_vari])

    #相関行列
    st.subheader('●相関行列')

    #相関行列
    vari_corr=df[desc_vari].corr()
    st.table(vari_corr)

    #相関行列のヒートマップ
    st.subheader('●相関のヒートマップ ')
    fig_corr, ax_corr = plt.subplots(figsize=(12, 9))
    sns.heatmap(vari_corr, square=True, vmax=1, vmin=-1, center=0)
    st.write(fig_corr)

    #散布図の制約条件
    #st.write('Constraints')
    #add_constraint=st.button('Add constraints')


  #ヒストグラムのページ
  if add_selectbox=='ヒストグラム':
    st.header('ヒストグラム')
    #表示変数の選択
    hist_vari=st.selectbox('表示変数',(df.columns.values))

    fig_hist = px.histogram(df[hist_vari], x=hist_vari, histnorm='probability')
    skew_info=df[hist_vari].skew()
    kurt_info=df[hist_vari].kurt()
    data_skew_kurt=pd.DataFrame({'尖度':[skew_info],'歪度':[kurt_info]})

    st.write(fig_hist)
    st.table(data_skew_kurt)

  #####折れ線グラフ・棒グラフ#####
  if add_selectbox=='折れ線グラフ・棒グラフ':
    #表示変数の選択
    st.header('折れ線グラフ・棒グラフ')

    disp_vari=choose_some_vari(df)

    fix_left, fix_right=st.beta_columns(2)
    #X軸を指定する
    x_fix=fix_left.checkbox('x軸を指定')

    st.write('●折れ線グラフ')

    if x_fix:

      #差を取得
      graph_vari=fix_right.selectbox('X軸の値',(set(df.columns.values) - set(disp_vari)))
      df_fix = df.set_index(graph_vari, drop=False)

      st.line_chart(df_fix[disp_vari])

    else:
      st.line_chart(df[disp_vari])


    #表示変数の選択
    st.write('●棒グラフ')
    bar2_vari=st.selectbox('縦軸の指定',(df.columns.values))

    #横軸の指定
    attributes=st.selectbox('横軸の指定',(df.columns.values))
    fig_bar2= go.Figure(data=[go.Bar(name='棒グラフ', x=df[attributes], y=df[bar2_vari])])
    fig_bar2.update_layout(xaxis_title=attributes,yaxis_title=bar2_vari,autosize=False,width=1024,height=768)

    st.write(fig_bar2)

  #####2変数の比較#####
  if add_selectbox=='2変数の関係':
    #相関図の作成
    st.subheader('●２変数の相関図')
    #相関図に表示する変数の選択
    dotmap_vari=choose_two_vari(df)
    #散布図の作成
    fig_dot = px.scatter(x=df[dotmap_vari[0]], y=df[dotmap_vari[1]])
    st.plotly_chart(fig_dot, use_container_width=True)


  #####散布図の表示
  if add_selectbox=='群間比較':
    st.header('群間比較')
    #散布図の表示

    #ペアプロット
    st.subheader('●ペアプロット')
    pair_compare=st.selectbox('2つの比較対照群を選択',(df.columns.values))
    fig_pairplot=sns.pairplot(df, hue=pair_compare)
    st.pyplot(fig_pairplot)

  #回帰分析のページ
  if add_selectbox=='回帰分析':
    st.header('回帰分析')
    #レイアウトを２分割
    left_column, right_column=st.beta_columns(2)
    Depe_value=left_column.selectbox('被説明変数',(df.columns.values))
    Inde_values=right_column.multiselect('説明変数',(df.columns.values))

    #回帰分析
    clf = LinearRegression()
    # 予測モデルを作成
    clf.fit(df[Inde_values], df[Depe_value])
    linear_graph=plt.scatter(df[Inde_values], df[Depe_value])
    linear_graph.plot(df[Inde_values], clf.predict(Depe_value))
    st.write(linear_graph)

  #クラスタリング のページ
  if add_selectbox=='クラスタリング':
    st.header('クラスタリング')
    st.write('準備中')

  #機械学習のページ
  if add_selectbox=='機械学習':
    st.header('機械学習')

else:

  #st.write('Upload a file to start your science life!!')
  image = Image.open('eniguma.jpeg')
  st.image(image, caption='',use_column_width=True)
