from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

prompts = [
    "介绍一下快手这家公司",
    "今天周几，你是怎么判断的。",
    "根据下面的文本分析一下流浪的含义：《流浪地球》是一部2019年上映的中国科幻电影，由郭帆执导，基于刘慈欣的同名小说改编。这部电影讲述了在未来几十年内，太阳即将成为红巨星，威胁到地球的生命。为了拯救地球，人类决定采取大胆的计划，将地球推出太阳系，进入另一个恒星系寻找新家园。电影中展现了人类面临的困境和挑战，以及科技与人性的冲突与融合。这部电影在中国内地和海外市场都取得了巨大的成功，被誉为中国科幻电影的里程碑之作。",  # noqa
    "为一段探索爵士乐的历史和文化意义的YouTube视频编写脚本。",
    "这周只工作了两天, 没有什么进展, 给我写一份工作周报, 要体现我充实的工作.",
    "撰写一篇有趣的旅行博客文章，介绍最近去夏威夷的旅行经历，重点突出文化体验和必看景点。",
    "写一篇交响乐音乐会评论，讨论乐团的表现和整体观众体验。",
    "使用适当的格式来构建一封正式的推荐信，为一名申请计算机科学研究生项目的学生提供推荐。",
    "起草一封引人入胜的产品发布公告电子邮件，通知我们的客户我们的新软件解决方案。",
    "起草一封致歉邮件，向一位经历了订单延迟的客户道歉，并保证问题已得到解决。",
    "对以下信息整理成为一段流畅的一段短视频拍摄脚本内容：产品名称韩伦美额头贴、多个美女、衰老变丑痛点、不安全痛点、性价比高、用料好、视频开头体现卖点、有数字定量描述、额外赠品/福利、包退包赔、包邮包送、好评度高、使用排比手法、对比反差，行动号召购买、拍摄体现使用教程，短视频时长在60s-90s",  # noqa
    "据输入提取关键词:苹果配件全家桶，让你体验磁吸无线充的快乐！关键才200就可拿下！# 苹果配件#手机配件#数码产品 华强北六件套全新升级，理想好货 买到就是赚到！ 苹果六件套全新升级，理想好货，买到就是赚到！#苹果配件/数码配件 新升级六件套，这一套你想要的它都有，只要200轻松拿下#苹果配件/数码产品 好货买到就是赚到！ 苹果配件全新升级，超值好货，买到就是赚到！ 理想好货，买到就是赚到！关键词:",  # noqa
    "苹果配件全家桶，让你体验磁吸无线充的快乐！关键才200就可拿下！# 苹果配件#手机配件#数码产品 华强北六件套全新升级，理想好货 买到就是赚到！ 苹果六件套全新升级，理想好货，买到就是赚到！#苹果配件/数码配件 新升级六件套，这一套你想要的它都有，只要200轻松拿下#苹果配件/数码产品 好货买到就是赚到！ 苹果配件全新升级，超值好货，买到就是赚到！ 理想好货，买到就是赚到！提取上文关键词：",  # noqa
]

model_dict = {
    "THUDM/chatglm-6b": {"model": AutoModel},
    "FreedomIntelligence/phoenix-inst-chat-7b": {"model": AutoModelForCausalLM},
}
# model_name = "THUDM/chatglm-6b"
model_name = "FreedomIntelligence/phoenix-inst-chat-7b"

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name], trust_remote_code=True)
    model = model_dict[model_name]["model"].from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir='/data/zhaoliangxuan/model_zoo'
    ).half().cuda()
    model = model.eval()
    for prompt in prompts:
        response, history = model.chat(tokenizer, prompt, history=[])
        print('=' * 20)
        print("问题：", prompt)
        print("回答：", response)
        print()
