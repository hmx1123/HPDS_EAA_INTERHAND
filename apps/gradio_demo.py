import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from core.test_utils import InterRender
import cv2
import einops
import requests
from pathlib import Path
import gradio as gr
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from utils.utils import get_mano_path, imgUtils
import cv2 as cv

model = InterRender(cfg_path='misc/model/config.yaml',
                    model_path="output/model/exp/hpds_eaa_ppp_best.pth",
                    render_size=256)


def process(img_path):

    with torch.no_grad():
        img = cv.imread(img_path)
        img = imgUtils.pad2squre(img)
        img = cv.resize(img, (256, 256))
        params, _, _, _, otherInfo = model.run_model(img)

        img_overlap = model.render(params, bg_img=img)

        if img.shape[0] != img_overlap.shape[0]:
            # 计算缩放比例，保持宽高比
            scale = img.shape[0] / img_overlap.shape[0]
            new_width = int(img_overlap.shape[1] * scale)
            img_overlap = cv.resize(
                img_overlap, (new_width, img.shape[0]))

    return [img_overlap]


# def save_img(save_path, result_gallery):
#     # [save_path, result_gallery] = save_ips
#     if save_path != '':
#         img_name = os.path.basename(result_gallery[0]['name'])
#         img_name = img_name[:img_name.find('.')]
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
        
#         img_path = result_gallery[0]['data']
#         img = gr.FileData(img_path)

#         cv.imwrite(os.path.join(save_path,
#                                 img_name + '_output_img.jpg'), img)
#         return 'Saved'
#     else:
#         return 'Please input save path!'


theme = gr.themes.Soft(primary_hue="blue")
block = gr.Blocks(
    theme=theme, title="InterHand 3D Mesh Reconstruction", show_api=False).queue()
with block:
    with gr.Row():
        gr.Markdown("## Inter Hand 3D Mesh Reconstruction")
    with gr.Row():
        with gr.Column():
            img_path = gr.Image(source='upload', type="filepath")
            run_button = gr.Button("Run", variant="primary")

        with gr.Column():
            result_gallery = gr.Gallery(
                label='Output', show_label=True, elem_id="gallery", columns=1, height='auto')
            # with gr.Group(label="SAVE"):
            #     with gr.Row():
            #         save_path = gr.Textbox(label="Save Path", min_width=450,
            #                                container=False, value=None, placeholder="Enter the save path")
                    # save_button = gr.Button("Save", min_width=50, variant="primary")
                # with gr.Row():
                #     img_overlap_checkbox = gr.Checkbox(
                #         label="img_overlap", value=True)

    ips = [img_path]
    # save_ips = [save_path, result_gallery]

    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
    # save_button.click(fn=save_img, inputs=save_ips)


block.launch(server_name='127.0.0.1', server_port=7861, share=True)
