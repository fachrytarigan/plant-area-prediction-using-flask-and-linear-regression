from flask import Flask, redirect, url_for, render_template, request, flash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from sklearn import linear_model
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
import pickle
import base64
from io import BytesIO

import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = 'random string'

model_width_filename = 'width_predict_model.ml'
model_height_filename = 'height_predict_model.ml'
model_non_zero_filename = 'non_zero_predict_model.ml'

model_width = pickle.load(open(model_width_filename, 'rb'))
model_height = pickle.load(open(model_height_filename, 'rb'))
model_non_zero = pickle.load(open(model_non_zero_filename, 'rb'))


@app.route('/', methods=['POST', 'GET'])
def index():

    return render_template("index.html")


@app.route('/predict',  methods=['POST'])
def predict():
    if request.method == 'POST':
        suhu_air = float(request.form['suhu_air'])
        tds = float(request.form['tds'])

        features = [[suhu_air, tds]]

        width_predict = model_width.predict(features)
        height_predict = model_height.predict(features)
        non_zero_predict = model_non_zero.predict(features)

        import math
        output_width = math.trunc(width_predict[0])
        output_height = math.trunc(height_predict[0])
        output_non_zero = math.trunc(non_zero_predict[0])

        from datetime import datetime
        namafile = datetime.now().strftime("%Y%m%d-%H%M%S")
        width_predict_image = visual_width(suhu_air, tds, namafile)
        height_predict_image = visual_height(suhu_air, tds, namafile)
        non_zero_predict_image = visual_nonzero(suhu_air, tds, namafile)

        context = {
            "suhu_air": suhu_air,
            "tds": tds,
            "width_predict_image" : width_predict_image,
            "height_predict_image" : height_predict_image,
            "non_zero_predict_image": non_zero_predict_image
        }

        return render_template("predict.html", context=context)

    else:
        url_for('index')


def visual_width(suhu_air, tds, namafile):
    df = pd.read_csv('data_foto_tanaman.csv')
    df = df.set_index('image_name')

    X = df[['suhu_air', 'tds']].values.reshape(-1, 2)
    Y_WIDTH = df['width']

    ######################## Prepare model data point for visualization ###############################
    x = X[:, 0] #Suhu Air
    y = X[:, 1] #TDS
    z = Y_WIDTH #Width

    x_pred_biru = np.linspace(x.min(), x.max(), len(x))   # range of suhu air
    y_pred_biru = np.linspace(y.min(), y.max(), len(y))  # range of tds

    suhu_air = float(suhu_air)
    tds = float(tds)
    
    x_pred_merah = [suhu_air]
    y_pred_merah = [tds]
    
    xx_pred_merah, yy_pred_merah = np.meshgrid([suhu_air], [tds])
    xx_pred_biru,yy_pred_biru = np.meshgrid(x_pred_biru, y_pred_biru)
    
    model_viz_biru = np.array([xx_pred_biru.flatten(), yy_pred_biru.flatten()]).T
    model_viz_merah = np.array([x_pred_merah, y_pred_merah]).T #Model Predicted
    
    ################################################ Train #############################################
    ols_width_biru = LinearRegression()
    model_width_biru = ols_width_biru.fit(X, Y_WIDTH)
    predicted_width_biru = model_width_biru.predict(model_viz_biru)
    
    ols_width_merah = LinearRegression()
    model_width_merah = ols_width_merah.fit(X, Y_WIDTH)
    predicted_width_merah = model_width_merah.predict(model_viz_merah)

    r2 = model_width_merah.score(X, Y_WIDTH)

    ############################################## Plot ################################################
    plt.style.use('seaborn-whitegrid')
    
    fig = Figure(figsize=(14, 4))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    axes = [ax1, ax2, ax3]
    for ax in axes:
   
        #ax.plot(x, y, z, color='#70b3f0', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(xx_pred_biru.flatten(), yy_pred_biru.flatten(), predicted_width_biru, facecolor=(0,0,0,0), s=20, edgecolor='#fffdd5', alpha=0.2)
        ax.scatter(x_pred_merah, y_pred_merah, predicted_width_merah, facecolor='#ff0000', s=20, edgecolor='#ff0000', marker='o', alpha=1)
        ax.plot(x, y, z, color='#2c6fff', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.set_xlabel('Suhu Air (°C)', fontsize=12)
        ax.set_ylabel('TDS (ppm)', fontsize=12)
        ax.set_zlabel('Leaf Area', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')

    # ax1.text2D(0.2, 0.32, 'Pertumbuhan Aquaponik', fontsize=13, ha='center', va='center',
    #         transform=ax1.transAxes, color='grey', alpha=0.5)
    # ax2.text2D(0.3, 0.42, 'Pertumbuhan Aquaponik', fontsize=13, ha='center', va='center',
    #         transform=ax2.transAxes, color='grey', alpha=0.5)
    # ax3.text2D(0.85, 0.85, 'Pertumbuhan Aquaponik', fontsize=13, ha='center', va='center',
    #         transform=ax3.transAxes, color='grey', alpha=0.5)

    ax1.view_init(elev=28, azim=120)
    ax2.view_init(elev=4, azim=114)
    ax3.view_init(elev=60, azim=165)

    #ax1.view_init(elev=27, azim=112)
    #ax2.view_init(elev=20, azim=-51)
    #ax3.view_init(elev=60, azim=165)

    fig.suptitle('Width Prediction = %f Pixels' % predicted_width_merah[0], fontsize=20)
    #fig.suptitle('$R^2 = %f$' % r2, fontsize=20)
    
    image_path = 'static/images/' + namafile + '-width.png'
    width_image = fig.savefig(image_path)
    fig.tight_layout()
    
    return image_path

def visual_height(suhu_air, tds, namafile):
    df = pd.read_csv('data_foto_tanaman.csv')
    df = df.set_index('image_name')

    X = df[['suhu_air', 'tds']].values.reshape(-1,2)
    Y_HEIGHT = df['height'] 

    ######################## Prepare model data point for visualization ###############################
    x = X[:, 0] #Suhu Air
    y = X[:, 1] #TDS
    z = Y_HEIGHT #Width

    x_pred_biru = np.linspace(x.min(), x.max(), len(x))   # range of suhu air
    y_pred_biru = np.linspace(y.min(), y.max(), len(y))  # range of tds

    suhu_air = float(suhu_air)
    tds = float(tds)
    
    x_pred_merah = [suhu_air]
    y_pred_merah = [tds]
    
    xx_pred_merah, yy_pred_merah = np.meshgrid([suhu_air], [tds])
    xx_pred_biru,yy_pred_biru = np.meshgrid(x_pred_biru, y_pred_biru)
    
    model_viz_biru = np.array([xx_pred_biru.flatten(), yy_pred_biru.flatten()]).T
    model_viz_merah = np.array([x_pred_merah, y_pred_merah]).T #Model Predicted
    
    ################################################ Train #############################################
    ols_height_biru = LinearRegression()
    model_height_biru = ols_height_biru.fit(X, Y_HEIGHT)
    predicted_height_biru = model_height_biru.predict(model_viz_biru)

    ols_height_merah = LinearRegression()
    model_height_merah = ols_height_merah.fit(X, Y_HEIGHT)
    predicted_height_merah = model_height_merah.predict(model_viz_merah)

    r2 = model_height_merah.score(X, Y_HEIGHT)

    ############################################## Plot ################################################
    plt.style.use('seaborn-whitegrid')
    
    fig = Figure(figsize=(14, 4))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    axes = [ax1, ax2, ax3]
    for ax in axes:
   
        #ax.plot(x, y, z, color='#70b3f0', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(xx_pred_biru.flatten(), yy_pred_biru.flatten(), predicted_height_biru, facecolor=(0,0,0,0), s=20, edgecolor='#fffdd5', alpha=0.2)
        ax.scatter(x_pred_merah, y_pred_merah, predicted_height_merah, facecolor='#ff0000', s=20, edgecolor='#ff0000', marker='o', alpha=1)
        ax.plot(x, y, z, color='#2c6fff', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.set_xlabel('Suhu Air (°C)', fontsize=12)
        ax.set_ylabel('TDS (ppm)', fontsize=12)
        ax.set_zlabel('Leaf Area', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')

    # ax1.text2D(0.2, 0.32, 'Pertumbuhan Aquaponik', fontsize=13, ha='center', va='center',
    #         transform=ax1.transAxes, color='grey', alpha=0.5)
    # ax2.text2D(0.3, 0.42, 'Pertumbuhan Aquaponik', fontsize=13, ha='center', va='center',
    #         transform=ax2.transAxes, color='grey', alpha=0.5)
    # ax3.text2D(0.85, 0.85, 'Pertumbuhan Aquaponik', fontsize=13, ha='center', va='center',
    #         transform=ax3.transAxes, color='grey', alpha=0.5)

    ax1.view_init(elev=28, azim=120)
    ax2.view_init(elev=4, azim=114)
    ax3.view_init(elev=60, azim=165)

    #ax1.view_init(elev=27, azim=112)
    #ax2.view_init(elev=20, azim=-51)
    #ax3.view_init(elev=60, azim=165)

    fig.suptitle('Height Prediction = %f Pixels' % predicted_height_merah[0], fontsize=20)
    #fig.suptitle('$R^2 = %f$' % r2, fontsize=20)
    
    image_path = 'static/images/' + namafile + '-height.png'
    width_image = fig.savefig(image_path)
    fig.tight_layout()

    return image_path    

def visual_nonzero(suhu_air, tds, namafile):
    df = pd.read_csv('data_foto_tanaman.csv')
    df = df.set_index('image_name')

    X = df[['suhu_air', 'tds']].values.reshape(-1,2)
    Y_NON_ZERO = df['non_zero'] 

    ######################## Prepare model data point for visualization ###############################
    x = X[:, 0] #Suhu Air
    y = X[:, 1] #TDS
    z = Y_NON_ZERO #Width

    x_pred_biru = np.linspace(x.min(), x.max(), len(x))   # range of suhu air
    y_pred_biru = np.linspace(y.min(), y.max(), len(y))  # range of tds

    suhu_air = float(suhu_air)
    tds = float(tds)
    
    x_pred_merah = [suhu_air]
    y_pred_merah = [tds]
    
    xx_pred_merah, yy_pred_merah = np.meshgrid([suhu_air], [tds])
    xx_pred_biru,yy_pred_biru = np.meshgrid(x_pred_biru, y_pred_biru)
    
    model_viz_biru = np.array([xx_pred_biru.flatten(), yy_pred_biru.flatten()]).T
    model_viz_merah = np.array([x_pred_merah, y_pred_merah]).T #Model Predicted
    
    ################################################ Train #############################################
    ols_non_zero_biru = LinearRegression()
    model_non_zero_biru = ols_non_zero_biru.fit(X, Y_NON_ZERO)
    predicted_non_zero_biru = model_non_zero_biru.predict(model_viz_biru)

    ols_non_zero_merah = LinearRegression()
    model_non_zero_merah = ols_non_zero_merah.fit(X, Y_NON_ZERO)
    predicted_non_zero_merah = model_non_zero_merah.predict(model_viz_merah)

    r2 = model_non_zero_merah.score(X, Y_NON_ZERO)

    ############################################## Plot ################################################
    plt.style.use('seaborn-whitegrid')
    
    fig = Figure(figsize=(14, 4))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    axes = [ax1, ax2, ax3]
    for ax in axes:
   
        #ax.plot(x, y, z, color='#70b3f0', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(xx_pred_biru.flatten(), yy_pred_biru.flatten(), predicted_non_zero_biru, facecolor=(0,0,0,0), s=20, edgecolor='#fffdd5', alpha=0.2)
        ax.scatter(x_pred_merah, y_pred_merah, predicted_non_zero_merah, facecolor='#ff0000', s=20, edgecolor='#ff0000', marker='o', alpha=1)
        ax.plot(x, y, z, color='#2c6fff', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.set_xlabel('Suhu Air (°C)', fontsize=12)
        ax.set_ylabel('TDS (ppm)', fontsize=12)
        ax.set_zlabel('Leaf Area', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')

    # ax1.text2D(0.2, 0.32, 'Pertumbuhan Aquaponik', fontsize=13, ha='center', va='center',
    #         transform=ax1.transAxes, color='grey', alpha=0.5)
    # ax2.text2D(0.3, 0.42, 'Pertumbuhan Aquaponik', fontsize=13, ha='center', va='center',
    #         transform=ax2.transAxes, color='grey', alpha=0.5)
    # ax3.text2D(0.85, 0.85, 'Pertumbuhan Aquaponik', fontsize=13, ha='center', va='center',
    #         transform=ax3.transAxes, color='grey', alpha=0.5)

    ax1.view_init(elev=28, azim=120)
    ax2.view_init(elev=4, azim=114)
    ax3.view_init(elev=60, azim=165)

    #ax1.view_init(elev=27, azim=112)
    #ax2.view_init(elev=20, azim=-51)
    #ax3.view_init(elev=60, azim=165)

    fig.suptitle('Leaf Area Prediction = %f' % predicted_non_zero_merah[0], fontsize=20)
    #fig.suptitle('$R^2 = %f$' % r2, fontsize=20)
    
    image_path = 'static/images/' + namafile + '-non-zero.png'
    width_image = fig.savefig(image_path)
    fig.tight_layout()

    return image_path

if __name__ == '__main__':
    app.run(debug=True)
