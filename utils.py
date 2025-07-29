from IPython.display import HTML, display
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np


def display_video(video_path):
    """
    Display a video in a notebook.
    
    Parameters:
    video_path (str): Path to the video file.
    """
    
    display(HTML(f"""
    <video controls width="600">
    <source src="{video_path}" type="video/mp4">
    Your browser does not support the video tag.
    </video>
    """))

def image_to_base64(img):
    """Convert a NumPy image to base64 PNG string."""
    buf = BytesIO()
    plt.imsave(buf, img, format='png', cmap='gray' if img.ndim == 2 else None)
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return base64_img

def display_image_grid_html(grid, titles, best_idx=None):
    M, N = grid.shape

    html = '<div style="display: flex; flex-direction: column; background-color: #dddddd; border: 2px solid #cccccc; border-radius: 10px; padding: 10px; padding-bottom: 20px;">'
    html += f'<h2 style="text-align: center; color: #333333;">Detected Faces</h2>'
    html += f'<table style="border-collapse: collapse;">'
    for row in range(N):
        if best_idx is not None and row == best_idx:
            line_style = "text-align: center; font-weight: bold; border: 3px solid #66ad39;"
        else:
            line_style = "text-align: center; font-weight: bold;"

        html += f'<tr style="{line_style}">'
        for col in range(M):
            idx = row * M + col
            img = np.array(grid[col, row])
            img_b64 = image_to_base64(img)
            img_size = 100 if M > 5 else 200
            html += f'<td style="padding: 0.3%; background-color: #dddddd;"><img src="data:image/png;base64,{img_b64}" ></td>'
        html += '</tr>'
        if best_idx is not None and row == best_idx:
            style = "color: #4a8723;"
        else:
            style = "color: #888888;"
        html += f'<tr style="text-align: center; font-weight: bold; background-color: #dddddd;"><td colspan="{M}" style="{style}">Speaking probability: {titles[row]}</td></tr>'
        
        if row < N - 1:
            html += f'<tr style="text-align: center;"><td colspan="{M}"><hr style="border: 1px solid #cccccc; width: 10%; margin-bottom: 0.8%;"></td></tr>'

    html += '</table> </div>'
    display(HTML(html))


def user_memory_to_html(memory_user, user_image, legend):
    """
    Convert user memory to a nice html display format.
    """
    html = '<div style="display: flex; background-color: #dddddd; border: 2px solid #cccccc; border-radius: 10px; padding: 10px; padding-bottom: 20px; flex-direction: column; align-items: center; width: 80%; margin: auto;">'
    html += '<h2 style="color: #333333; text-align: center; width: 100%; margin:0; padding:0; margin-top: 10px; margin-bottom: 10px;">Detected User</h2>'
    html += '<div style="display: flex; align-items: flex-start; gap: 20px; width: 100%;">'

    #user image

    html += f"<div style='display: flex; flex-direction: column; align-items: center; width: 30%; align-self: center;'>"
    img64 = image_to_base64(user_image)
    html += f"<img src='data:image/png;base64,{img64}' alt='User Image' style='max-width: 50%; height: auto; display: block; margin-bottom: 10px;'>"
    html += f"<div style='font-size: 16px; font-weight: bold; color: #888888;'>{legend}</div> </div>"

    #user memory

    html += "<div style='width: 60%; margin-left: 20px; align-self: center; max-height: 500px; overflow-y: auto;'>"
    html += "<h3 style='padding-left: 2%; color: #888888;'>User Memory:</h3>"
    html += "<table style='border-collapse: collapse; background-color: #eeeeee; width: 100%;'>"
    html += (
        "<tr style='background-color: #bbbbbb; text-align: center;'>"
        "<th style='width: 30%; color: #ffffff; padding: 5px; text-align: center; font-weight: bold;'>"
        "Feature"
        "</th>"
        "<th style='color: #ffffff; padding: 5px; text-align: center; font-weight: bold; width: 70%;'>"
        "Value"
        "</th>"
        "</tr>"
    )
    primary_features = [feature for feature in memory_user if feature['type'] == 'primary']
    context_features = [feature for feature in memory_user if feature['type'] == 'contextual']

    for feature in primary_features:
        content = feature['value'] if feature['value'] else "N/A"
        html += f"<tr style='text-align: center;'><td style = 'width: 30%; color: #888888; font-weight: 600; padding: 5px;'>{feature['name']}</td><td style = 'color: #888888; padding: 5px;'>{content}</td></tr>"

    for i, feature in enumerate(context_features):
        if i == 0:
            style = "border-top: 3px solid #bbbbbb; text-align: center;"
        else:
            style = "text-align: center;"
        
        content = feature['value'] if feature['value'] else "N/A"
        html += (
            f"<tr style='{style}'>"
            f"<td style='width: 30%; color: #888888; padding: 5px;'>"
            f"<b style='font-weight: 600;'>{feature['name']}</b> <br/> <i style='color: #aaaaaa;'>({feature['description']})</i>"
            f"</td>"
            f"<td style='color: #888888; padding: 5px;'>"
            f"{content}"
            f"</td>"
            f"</tr>\n"
        )

    html += "</table>"
    html += "</div> </div>"

    display(HTML(html))
