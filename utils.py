from IPython.display import HTML, display
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import cv2


# === Utility ===

def image_to_base64(img):
    """Convert a NumPy image to base64 PNG string."""
    buf = BytesIO()
    plt.imsave(buf, img, format='png', cmap='gray' if img.ndim == 2 else None)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# === Shared HTML snippets ===

def style_box(title, width='100%', bg_color='#dddddd'):
    return f'''
    <div style="display: flex; background-color: {bg_color}; border: 2px solid #cccccc;
        border-radius: 10px; padding: 10px 10px 20px 10px; flex-direction: column;
        align-items: center; width: {width}; margin: auto; margin-bottom: 20px;">
        <h2 style="color: #333333; text-align: center;">{title}</h2>
    '''

def feature_table_header():
    return """
    <table style='border-collapse: collapse; background-color: #eeeeee; width: 100%;'>
    <tr style='background-color: #bbbbbb; text-align: center;'>
        <th style='width: 30%; color: #ffffff; padding: 5px;'>Feature</th>
        <th style='width: 70%; color: #ffffff; padding: 5px;'>Value</th>
    </tr>
    """


# === Visuals ===

def display_video(video_path, jupyter=True):
    if jupyter:
        bg_color = '#dddddd'
    else:
        bg_color = '#f8f8f8'

    html = style_box("Input Video", '60%', bg_color)
    html += f"""
    <video controls width="600">
        <source src="{video_path}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    </div>
    """
    return html

def display_image_grid_html(grid, titles, best_idx=None, jupyter=True):
    if jupyter:
        bg_color = '#dddddd'
    else:
        bg_color = '#f8f8f8'

    M, N = grid.shape
    html = style_box("Detected Faces", '95%', bg_color)
    html += '<table style="border-collapse: collapse; width: 100%; box-sizing: border-box; margin:0">'

    for row in range(N):
        highlight = "border: 3px solid #66ad39;" if best_idx == row else ""
        html += f'<tr style="text-align: center; font-weight: bold; width:100%; {highlight}">'
        for col in range(min(18, M)):
            #blur all the images
            img= np.array(grid[col, row].copy())
            img_b64 = image_to_base64(np.array(img))
            html += f'<td style="padding: 0.3%;"><img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto;"></td>'
        html += '</tr>'

        color = "#4a8723" if best_idx == row else "#888888"
        html += f'<tr style="text-align: center;"><td colspan="{M}" style="color: {color}; font-weight: bold;">Speaking probability: {float(titles[row]):.2f}</td></tr>'

        if row < N - 1:
            html += f'<tr><td colspan="{M}"><hr style="border: 1px solid #ccc; width: 10%; margin-bottom: 0.8%;"></td></tr>'

    html += '</table></div>'
    return html


def user_memory_to_html(memory_user, user_image, legend, title="Detected User", jupyter=True):
    if jupyter:
        bg_color = '#dddddd'
    else:
        bg_color = '#f8f8f8'

    html = style_box(title, '80%', bg_color)
    html += '<div style="display: flex; align-items: flex-start; gap: 20px; width: 100%;">'

    # Left: User image
    img64 = image_to_base64(user_image)
    html += f"""
    <div style='width: 30%; text-align: center; align-self: center;'>
        <img src='data:image/png;base64,{img64}' style='max-width: 50%; height: auto; margin-bottom: 10px;'>
        <div style='font-size: 16px; font-weight: bold; color: #888;'>{legend}</div>
    </div>
    """

    # Right: Memory table
    html += "<div style='width: 60%; overflow-y: auto;'>"
    html += "<h3 style='color: #888;'>User Memory:</h3>"
    html += feature_table_header()

    for feature in memory_user:
        content = feature.get('value', "N/A") or "N/A"
        if feature["type"] == "primary":
            html += f"<tr style='text-align: center;'><td style='color: #888; font-weight: 600; padding: 5px;'>{feature['name']}</td><td style='color: #888; padding: 5px;'>{content}</td></tr>"
    
    for i, feature in enumerate([f for f in memory_user if f['type'] == 'contextual']):
        border = "border-top: 3px solid #bbbbbb;" if i == 0 else ""
        html += f"""
        <tr style='text-align: center; {border}'>
            <td style='color: #888; padding: 5px;'><b>{feature['name']}</b><br/><i style='color: #aaa;'>({feature['description']})</i></td>
            <td style='color: #888; padding: 5px;'>{feature.get('value', 'N/A')}</td>
        </tr>
        """

    html += "</table></div></div></div>"
    return html


def display_pie_chart(labels, values, emotion, prob, jupyter=True):
    if jupyter:
        bg_color = '#dddddd'
    else:
        bg_color = '#f8f8f8'

    html = style_box("Emotion Detection", '50%', bg_color)
    html += f"""
    <div style="display: flex; gap: 20px; width: 100%; flex-direction: row; align-items: center;">
        <div style="text-align: center; width: 40%; align-self: center;">
            <h3 style="color: #333;">Detected Emotion: <span style="margin-left: 5px; border: 2px solid limegreen; padding: 5px;">{emotion.capitalize()}</span></h3>
            <p style="color: #555;">Probability: {prob:.2f}</p>
        </div>
        <div style="width: 50%; justify-content: center;">
    """
    fig = go.Figure([go.Pie(labels=labels, values=values, textinfo='percent+label', showlegend=False)])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    html += fig.to_html(full_html=False, include_plotlyjs='cdn') + "</div></div></div>"
    return html


def display_sequence_with_transcription(image_row, transcription, jupyter=True):
    if jupyter:
        bg_color = '#dddddd'
    else:
        bg_color = '#f8f8f8'

    html = style_box("Audio Transcription", "95%", bg_color)
    html += '<table style="border-collapse: collapse; box-sizing: border-box; width: 100%;"><tr>'
    for img in image_row:
        
        img_base64 = image_to_base64(np.array(img))
        html += f'<td><img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;"></td>'
    html += f'</tr></table><p style="color: #555; margin-top: 10px; font-size:18px">"{transcription.strip()}"</p></div>'
    return html


def display_answer(answer, memory_user, retrieved_features_names, generation_time, jupyter=True):
    if jupyter:
        bg_color = '#dddddd'
    else:
        bg_color = '#f8f8f8'

    html = style_box("Generated answer:", '80%', bg_color)
    html += f'<div style="font-size: 16px; font-weight: 500; color: #888; width: 60%; text-align: center; margin: 15px;">"{answer}"</div>'
    html += f"""
    <div style='display: flex; justify-content: center; align-items: center; width:100%'>
        <div style='font-size: 18px; color: #888; width: 30%; text-align: center;'>Generation time: {generation_time:.2f} seconds</div>
        <div style='width: 60%; max-height: 500px; overflow-y: auto;'>
            <h3 style='color: #888;'>Retrieved Features:</h3>
            {feature_table_header()}
    """

    for i, feature in enumerate([f for f in memory_user if f["name"] in retrieved_features_names]):
        border = "border-top: 3px solid #bbbbbb;" if i == 0 else ""
        description = f"<br/><i style='color: #aaa;'>({feature['description']})</i>" if feature["type"] == "contextual" else ""
        content = feature.get("value", "N/A") or "N/A"
        html += f"""
        <tr style='text-align: center; {border}'>
            <td style='color: #888; padding: 5px;'><b>{feature['name']}</b>{description}</td>
            <td style='color: #888; padding: 5px;'>{content}</td>
        </tr>
        """

    html += "</table></div></div></div>"
    return html

def save_html_page(html_blocks, filename="output.html"):
    """Save a list of HTML blocks to a single HTML file."""
    style = """
    <style>
    body { font-family: Arial, sans-serif; background-color: #dddddd; color: #333; }
    </style>
    """

    with open(filename, 'w') as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Output</title>{style}</head><body>")
        for block in html_blocks:
            f.write(str(block))
        f.write("</body></html>")