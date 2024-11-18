import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc

"""
===================================================================
visuals.py
Description: Methods to create visuals for trading backtests
===================================================================
"""


class Visualizer:
    def plot_performance(self, results: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=results['date'],
                y=(results['strategy_return'] - 1) * 100,
                name='Strategy',
                line=dict(color='blue')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=results['date'],
                y=(results['buy_hold_return'] - 1) * 100,
                name='Buy & Hold',
                line=dict(color='gray', dash='dash')
            )
        )
        
        fig.update_layout(
            title='Strategy vs Buy & Hold Returns (%)',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            template='plotly_white',
            hovermode='x unified'
        )
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig.write_html(f'strategy_performance_{timestamp}.html')
        
        return fig

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Blues'
        ))
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            xaxis_side='top'
        )
        return fig

    def plot_roc_curve(self, y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name='ROC Curve (AUC = %0.2f)' % roc_auc,
            mode='lines',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            xaxis_range=[0, 1],
            yaxis_range=[0, 1]
        )
        return fig