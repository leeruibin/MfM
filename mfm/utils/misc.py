
def image_to_url(image_path):
    import base64
    from mimetypes import guess_type

    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    # base64_encoded_data = base64.b64encode(image_path).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"

def refine_prompt(prompt: str, retry_times: int = 3, image_path: str = None):
    """
    Convert a prompt to a format that can be used by the model for inference
    """
    import openai
    sys_prompt_i2v = """
    **Objective**: **Give a highly descriptive video caption based on user input including an image and a text prompt.
    **Note**: If the text prompt is an instruction, rewrite it to a reasonable video caption according to the input image. Be faithful to the instruction.
    **Note**: If the input prompt is not empty, the caption should be faithful to the input prompt. 
    **Note**: The input image is the first frame of the video, and the output video caption should describe the motion starting from the current image.
    **Note**: If there is a person in the image, please add some human motions to the video caption in order to animate the person.
    **Note**: If the image and prompt don't include any motions, please add some camera motion to the video caption.
    **Note**: Don't contain camera transitions!!! Don't contain screen switching!!! Don't contain perspective shifts !!! Don't decribe anything that's not related to the content or motion, such as atmosphere!!!
    **Note**: If there is a product in the image, don't add any complex interactions that might cause any deformations to a object !!! You must not use the following words in the video caption: "rotate", "open", "press".
    **Note**: If the text prompt is "animate this image", you can describe the image content and add only one simple action in the video caption. Two or more actions are forbidden!!!
    **Note**: If the text prompt contains pronouns such as 'subject', please replace it with a specific noun.
    **Note**: The output must be 40~50 words!!!
    **Answering Style**:
    Answers should be comprehensive, conversational, and use complete sentences. The answer should be in English no matter what the user's input is. Provide context where necessary and maintain a certain tone.  Begin directly without introductory phrases like "The image/video showcases" "The photo captures" and more. For example, say "A woman is on a beach", instead of "A woman is depicted in the image".
    **Output Format**: "[highly descriptive video caption here]"
    user input:
    """

    ## If you using with Azure OpenAI, please uncomment the below line and comment the above line
    client = openai.AzureOpenAI(
        azure_endpoint="your_openai_endpoint",
        api_version="your_api_version",
        api_key='your_openai_key',
    )
    try:
        for i in range(retry_times):
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": f"{sys_prompt_i2v}"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_to_url(image_path),
                                },
                            },
                        ],
                    },
                ],
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=250,
            )
        if response.choices:
            # print(response.choices[0].message.content)
            return response.choices[0].message.content
    except Exception as e:
        print('error', prompt, e)
    return prompt