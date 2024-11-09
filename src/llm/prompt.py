def LLM_prompt_enhance(model_info, classification_result, original_prompt,logging):
    """
    根据模型的分类或异常检测结果增强 LLM 的提示词。

    Args:
        model_info (dict): 包含模型实例和类型的字典，例如 {'model': model, 'type': 'supervised'}。
        classification_result (dict): 分类或异常检测结果。
        original_prompt (str): 原始的 LLM 提示词。

    Returns:
        str: 增强后的 LLM 提示词。
    """
    method_type = model_info['type']

    try:
        if method_type == 'supervised':
            label = classification_result.get('predicted_label', 'Unknown')
            probability = classification_result.get('probability', 0.0)
            enhanced_prompt = f"{original_prompt}\n\n检测到的图像类别为: {label}，置信度为: {probability*100:.2f}%。"
            logging.info(f"增强后的提示词 (Supervised)正在使用{model_info['model']}进行检测: {enhanced_prompt}")
            return enhanced_prompt

        elif method_type in ['unsupervised', 'vit_anomaly']:
            if method_type == 'vit_anomaly':
                anomaly_score = classification_result.get('anomaly_score', 0.0)
                enhanced_prompt = f"{original_prompt}\n\n检测到的异常得分为: {anomaly_score:.4f}。"
            else:
                recon_error = classification_result.get('reconstruction_error', 0.0)
                enhanced_prompt = f"{original_prompt}\n\n图像的重构误差为: {recon_error:.4f}。"

            logging.info(f"增强后的提示词 (Unsupervised): {enhanced_prompt}")
            return enhanced_prompt

        else:
            logging.error(f"不支持的模型类型: {method_type}")
            raise ValueError(f"Unsupported method type: {method_type}")

    except KeyError as e:
        logging.error(f"分类结果中缺少关键字段: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"增强提示词失败: {str(e)}")
        raise