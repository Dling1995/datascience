import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
import altair as alt

#####クラスの定義#####
class constraints:

  def __init__(self):
    self.cons_1,self.cons_2,self.cons_3,self.cons_4=st.beta_columns(4)

  #アクティベーションボタン
  def activate(self):
    #アクティベートボタン
    self.cons_1.checkbox('activate', value=False)

  #変数選択
  def variable(self,data):
    self.cons_2.selectbox('Set a variable',(data))

  #不等号の条件式
  def formula(self):
    self.cons_3.selectbox('Set a constraint',('','>','≧','＝','<','≦','≠'))

  #数字の選択
  def value(self):
    self.cons_4.number_input('Set a value',)

######関数の定義######
def choose_two_vari(data):
  vari_left, vari_right=st.beta_columns(2)
  vari1=vari_left.selectbox('第一変数',(data.columns.values))
  vari2=vari_right.selectbox('第二変数',(data.columns.values))

  return vari1, vari2

def choose_some_vari(data):
  vari_some=st.multiselect(
    '表示変数の選択',
    (data.columns.values))

  return vari_some


####################

#####画面表示#########

#タイトルを表示
st.title('Data Science Life')

#サイドバーを加える
# ファイルアップロード

uploaded_file = st.sidebar.file_uploader("Upload a file to start your science life!!", type=['xlsx','csv'])

if uploaded_file is not  None:

  add_selectbox = st.sidebar.selectbox(
    'Information',
    ('アップロードデータ','基本統計量','折れ線グラフ・棒グラフ','散布図','回帰分析','Clustering', 'Machine Learning')
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

      #差を取得したい
      graph_vari=fix_right.selectbox('X軸の値',(set(df.columns.values) - set(disp_vari)))
      df_fix = df.set_index(graph_vari, drop=False)

      st.line_chart(df_fix[disp_vari])

    else:
      st.line_chart(df[disp_vari])

    #表示変数の選択
    st.write('●棒グラフ(度数)')
    bar_vari=st.selectbox('比較する値',(df.columns.values))

    #属性ごとに見る
    attribute_fix=st.checkbox('属性ごとに比較')

    if attribute_fix:
      attributes=st.selectbox('属性ごとに作成',(df.columns.values))
      fig = alt.Chart(df).mark_bar(size=5).encode(
          x=bar_vari,
          y='count()',
          column=alt.Column(attributes)
      ).properties(
          width=150,
          height=150
      ).interactive()

    else:
      fig = alt.Chart(df).mark_bar(size=60).encode(
          x=bar_vari,
          y='count()',
      ).properties(
          width=300,
          height=300
      ).interactive()

    st.write(fig)

    #棒グラフの表示
    #st.write('●棒グラフ')
    #st.pyplot(df[desc_vari])
    #st.write(
    #px.bar(df[desc_vari], x='medal', y='count' ,title="sample figure"))

  #散布図の表示
  if add_selectbox=='散布図':
    st.header('散布図')
    #散布図の表示

    #散布図に表示する変数の選択

    dotmap_vari=choose_two_vari(df)
    #散布図の制約条件
    st.write('Constraints')
    add_constraint=st.button('Add constraints')

#制約条件の追加
    #if add_constraint:
      #constaintslist=constraints()
      #それぞれの変数をリストでまとめて格納
      #activation=constaintslist.activate()
      #variable=constaintslist.variable(df.columns.values)
      #form=constaintslist.formula()
      #value=constaintslist.value()
      #constaintslist_each=[activation,variable,form,value]
      #constaintslist_all.append(constaintslist_each)




    #散布図の作成
    fig = px.scatter(x=df[dotmap_vari[0]], y=df[dotmap_vari[1]])
    st.plotly_chart(fig, use_container_width=True)

  #回帰分析のページ
  if add_selectbox=='回帰分析':
    st.header('回帰分析')
    #レイアウトを２分割
    left_column, right_column=st.beta_columns(2)
    left_column.write('変数選択')
    left_column.write('2次式以上の変数')
    right_column.write('モデル式')


  #クラスタリング のページ
  if add_selectbox=='Clustering':
    st.header('クラスタリング ')

  #機械学習のページ
  if add_selectbox=='Machine Learning':
    st.header('Mavhine Learning')
else:
  #st.write('Upload a file to start your science life!!')
  image = Image.open('eniguma.jpeg')
  st.image(image, caption='',use_column_width=True)
