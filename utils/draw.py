import csv
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column
import numpy as np
output_file("comparison.html")
path = '/home/weich/project/OBAQ/quantized.pytorch/results'
def getlog(f_csv):
    res = [(row['epoch'], row['train_loss'], row['val_loss'], row['train_error1'], row['val_error1'], row['train_error5'], row['val_error5']) for row in f_csv]
    return np.array(res, dtype=np.float32).T

def draw_log(figures, epochs, records, legend_label, line_color):
    for i in range(len(figures)):
        figures[i].line(epochs, records[i], legend_label = legend_label, line_color = line_color)

f_uniform_qint8 = open(path + '/quantized_resnet18_cifar100/results.csv')
f_fp = open(path + '/resnet18_cifar100/results.csv')
f_bfp8 = open(path + '/quantized_resnet18_BFP_test/results.csv')
f_bfp24 = open(path + '/quantized_resnet18_BFP24_test/results.csv')
f_bfp4 = open(path + '/K_25/results.csv')
f_375 = open(path + '/WK_195/results.csv')
f_265 = open(path + '/K_188/results.csv')
f_4 = open(path + '/A8W4G8/results.csv')
f_3 = open(path + '/W_3/results.csv')
f_2 = open(path + '/W_2/results.csv')

f_random = open(path + '/random/results.csv')
f_A4W8G8 = open(path + '/quantized_resnet18_20/results.csv')
f_A8W4G8 = open(path + '/quantized_resnet18_40/results.csv')
f_A8W8G4 = open(path + '/A8W8G4/results.csv')
f_A8W8G6 = open(path + '/quantized_resnet18_60/results.csv')

f_avg2 = open(path + '/results_fixed/avg2/results.csv')
f_avg3 = open(path + '/results_fixed/avg3/results.csv')
f_avg4 = open(path + '/results_fixed/avg4/results.csv')
f_avg32 = open(path + '/results_fixed/fixed_bits_3.2/results.csv')
f_avg22 = open(path + '/results_fixed/fixed_bits_2.2/results.csv')

log_uniform_qint8 = getlog(csv.DictReader(f_uniform_qint8))
log_fp = getlog(csv.DictReader(f_fp))
log_bfp8 = getlog(csv.DictReader(f_bfp8))
log_bfp24 = getlog(csv.DictReader(f_bfp24))
log_bfp4 = getlog(csv.DictReader(f_bfp4))
log_random = getlog(csv.DictReader(f_random))
log_A4W8G8 = getlog(csv.DictReader(f_A4W8G8))
log_A8W4G8 = getlog(csv.DictReader(f_A8W4G8))
log_A8W8G4 = getlog(csv.DictReader(f_A8W8G4))
log_A8W8G6 = getlog(csv.DictReader(f_A8W8G6))

log_avg2 = getlog(csv.DictReader(f_avg2))
log_avg3 = getlog(csv.DictReader(f_avg3))
log_avg4 = getlog(csv.DictReader(f_avg4))
log_fix32 = getlog(csv.DictReader(f_avg32))
log_fix22 = getlog(csv.DictReader(f_avg22))


train_loss = figure(x_axis_label='epochs', y_axis_label = 'train loss', title = 'Train_loss', plot_width = 400, plot_height = 300)
val_loss = figure(x_axis_label='epochs', y_axis_label = 'val loss', title = 'val_loss',  plot_width = 400, plot_height = 300)
train_error = figure(x_axis_label='epochs', y_axis_label = 'train error', title = 'Train_error1',  plot_width = 400, plot_height = 300)
val_error = figure(x_axis_label='epochs', y_axis_label = 'val error', title = 'Val_error1',  plot_width = 400, plot_height = 300)
train_error5 = figure(x_axis_label='epochs', y_axis_label = 'train error5', title = 'Train_error5',  plot_width = 400, plot_height = 300)
val_error5 = figure(x_axis_label='epochs', y_axis_label = 'val error5', title = 'Val_error5',  plot_width = 400, plot_height = 300)

figures = [train_loss, val_loss, train_error, val_error, train_error5, val_error5]
draw_log(figures, log_avg2[0], log_avg2[1:], legend_label='2', line_color='red')
draw_log(figures, log_avg3[0], log_avg3[1:], legend_label='3', line_color='blue')
draw_log(figures, log_avg4[0], log_avg4[1:], legend_label='4', line_color='orange')
draw_log(figures, log_fix32[0], log_fix32[1:], legend_label='dynamic(3.2)', line_color='green')
draw_log(figures, log_fix22[0], log_fix22[1:], legend_label='dynamic(2.2)', line_color='purple')
# draw_log(figures, log_A4W8G8[0], log_A4W8G8[1:], legend_label='20', line_color='red')
# draw_log(figures, log_A8W4G8[0], log_A8W4G8[1:], legend_label='40', line_color='blue')
# draw_log(figures, log_A8W8G4[0], log_A8W8G4[1:], legend_label='A8_W8_G4', line_color='orange')
# draw_log(figures, log_A8W8G6[0], log_A8W8G6[1:], legend_label='60', line_color='green')


show(column(figures))