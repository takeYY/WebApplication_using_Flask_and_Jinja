# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
#from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC


# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)

# メッセージをランダムに表示するメソッド
def picked_up():
    messages = [
        "こんにちは、あなたはタイタニック号で生き残れるでしょうか？",
        "やあ！タイタニックの生存を予測するぜ",
        "もし、あなたがタイタニック号に乗ったらどうなるか知りたくないか？"
    ]
    # NumPy の random.choice で配列からランダムに取り出し
    return np.random.choice(messages)

# ダミー変数化するメソッド
def to_dummies(train, test):
  train_size = len(train)
  # trainデータとtestデータをダミー変数化のため結合
  train_test = train.append(test, ignore_index=True)
  train_test_size = len(train_test)
  # ダミー変数化
  dummy = pd.get_dummies(train_test)
  # 結合データをtrainとtestに分割
  dummied_train = dummy[0:train_size]
  dummied_test = dummy[train_size:train_test_size+1]
  return dummied_train, dummied_test

# データを追加
def add_data_frame(original_data, add_data):
  new_data = pd.concat([original_data, add_data], axis=1)
  return new_data

# 敬称列を追加
def add_name_row(data):
    other_list = []
    last_name = []
    for name in data:
        first, other = name.split(",")
        try:
            last_name = other.split(".")
        except:
            last = "NAN"
        other_list.append(last_name[0])
    df = pd.DataFrame({"Title": other_list})
    return df

def predict_titanic(test):
  train = pd.read_csv("titanic_train.csv")
  train_x = train.drop(["Survived"], axis=1)
  train_y = train["Survived"]
  test_x = test.copy()
  test_x["SibSp"] = test_x["SibSp"].astype(np.int64)
  test_x["Parch"] = test_x["Parch"].astype(np.int64)
  test_x["Fare"] = test_x["Fare"].astype(np.int64)
  test_x["Pclass"] = test_x["Pclass"].astype(np.int64)
  test_x["Age"] = test_x["Age"].astype(np.int64)

  # PassengerIdを除外
  train_x = train_x.drop(["PassengerId"], axis=1)

  # Ticket, Cabin, Embarkedを除外
  train_x = train_x.drop(["Ticket", "Cabin"], axis=1)

  # FareのNaNを修正
  fix_row = {"Fare": 0, "Age": 30}
  train_x = train_x.fillna(fix_row)

  # 新たな敬称列を作成
  new_train_df = add_name_row(train_x["Name"])
  new_test_df = add_name_row(test_x["Name"])

  # Name列を削除
  train_x = train_x.drop(["Name"], axis=1)
  test_x = test_x.drop(["Name"], axis=1)

  # 敬称列を追加
  train_x = add_data_frame(train_x, new_train_df)
  test_x = add_data_frame(test_x, new_test_df)

  # ダミー変数化
  train_x, test_x = to_dummies(train_x, test_x)

  # Sex_maleを削除
  train_x = train_x.drop(["Sex_male"], axis=1)
  test_x = test_x.drop(["Sex_male"], axis=1)

  x_array = np.array(train_x)
  y_array = np.array(train_y)
  x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.4, random_state=114)
  
  # 学習と精度評価
  rfc = RFC(random_state=114)
  rfc.fit(x_train, y_train)
  pred = rfc.predict(x_test)
  accuracy = accuracy_score(pred, y_test)

  # テストデータの予測
  pred_2 = rfc.predict(test_x)
  return accuracy, pred_2

# ここからウェブアプリケーション用のルーティングを記述
# index にアクセスしたときの処理


@app.route('/')
def index():
    title = "タイタニックの生存予測"
    message = picked_up()
    # index.html をレンダリングする
    return render_template('index.html',
                           message=message, title=title)

# /post にアクセスしたときの処理


@app.route('/post', methods=['GET', 'POST'])
def post():
    title = "こんにちは"
    if request.method == 'POST':
        # リクエストフォームから「名前」を取得して
        name = request.form['name']
        data = {}
        data["Pclass"] = int(request.form['PClass'])
        data["SibSp"] = int(request.form['SibSp'])
        data["Parch"] = int(request.form['Parch'])
        data["Age"] = int(request.form['Age'])
        if data["Pclass"] == 1:
          data["Fare"] = 500
        elif data["Pclass"] == 2:
          data["Fare"] = 250
        else:
          data["Fare"] = 50
        sex = request.form["Sex"]
        if sex == "nothing":
          data["Sex"] = "female"
        else:
          data["Sex"] = sex
        data["Name"] = "test," + request.form["Title"] + ".testDAO"
        Title = request.form["Title"]
        df = pd.DataFrame.from_dict(data, orient='index').T
        df = pd.DataFrame(df)
        accuracy, pred = predict_titanic(df)
        accuracy = round(accuracy, 4) * 100
        return render_template('index.html', name=name,accuracy=accuracy, pred=pred[0], data=data, Title=Title)
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.debug = True  # デバッグモード有効化
    app.run(host='0.0.0.0', port=8000)  # どこからでもアクセス可能に
