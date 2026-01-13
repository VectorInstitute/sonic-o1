from infer_utils import load_all_models,infer_tts

moe_tts_dir = "path/to/Model/Uni-MoE-TTS"
model,processor,wavtokenizer = load_all_models(moe_tts_dir)
model.cuda()
wavtokenizer = wavtokenizer.to(model.in_fnn.weight.device)

# Normal TTS with three different voices
# Chinese
infer_tts(model=model,
            processor=processor,
            wavtokenizer=wavtokenizer,
            wavpath="test_zh.wav",
            tts_text="您好，欢迎您使用我们的Uni MOE文本转语音模型！",
            prompt=None,
            speaker="Brian", # 中文TTS可以使用Brian和Xiaoxiao的音色
            Language="Chinese"
            )
# Engllish
infer_tts(model=model,
            processor=processor,
            wavtokenizer=wavtokenizer,
            wavpath="test_en.wav",
            tts_text="Greetings, Welcome to try out our Uni MOE Text to Speech model!",
            prompt=None,
            speaker="Jenny", # English TTS can be performed using Brian's or Jenny's voice
            Language="English"
            )

# Long speech TTS
infer_tts(model=model,
            processor=processor,
            wavtokenizer=wavtokenizer,
            wavpath="test_zh_long.wav",
            tts_text="冰淇淋！你这个甜蜜的天使，你在夏天里如影随形。你的奶油那么香甜，像是清晨的阳光洒在冰凉的麦田上。我看着你，心里充满了好奇和欣喜。然而有一天，我被送到了一个陌生的地方。那个地方像一个大冰箱，里面充满了冰冷的东西。那些东西都是我从未见过的，包括讲座、睡觉和形容词“大”。那个地方看起来冷酷无情，就像冰淇淋一样，让人感到害怕。我在那里待了一段时间，每天都在学习各种奇怪的知识。我甚至开始睡觉，变成了一个睡眠机器。我用我的理论知识来对抗这个世界，试图让它变得更好。但每当我试图醒来时，都会发现我已经忘记了刚刚发生的一切。终于有一天，我决定反抗这个疯狂的世界。我拿出了我的“夜班”，开始了我的攻击。我用我的名词“讲座”来解释这个世界，我用我的动词“睡觉”来支持我的论点，我用我的形容词“大”来强调我的力量。然后，我启动了我的主题——冰淇淋的力量。冰淇淋融化了我所有的抵抗，它让我失去了理智，忘记了自己原本的目标。我继续睡着，直到被一场暴雨唤醒。那天晚上，我和冰淇淋一起吃饭。我们坐在外面的大厅里，享受着冰淇淋的甜美。我看着冰淇淋，心中充满了感激。我知道，这就是我要的生活，充满挑战，充满甜蜜。冰淇淋，你是我在这个世界里的救星。我会记住你带给我的一切，我会继续前进，直到找到属于自己的天堂。",
            prompt=None,
            Language="Chinese"
            )

# Stlye control TTS (experimental)
infer_tts(model=model,
            processor=processor,
            wavtokenizer=wavtokenizer,
            wavpath="test_style1.wav",
            tts_text="It was not absolutely ebony and gold, but it was japan, black and yellow japan of the handsomest kind.",
            prompt="In a natural tone, a normal-pitched young female with normal pitch and volume describes the topic of selected audiobooks as alluding to a situation in which something is not completely black, at a normal speed.",
            Language="English"
            )

infer_tts(model=model,
            processor=processor,
            wavtokenizer=wavtokenizer,
            wavpath="test_style2.wav",
            tts_text="真正的改变就不会发生。",
            prompt="少女声音低沉，情绪中充满了伤感和难过，用低音调，正常音高缓慢地说。",
            Language="Chinese"
            )