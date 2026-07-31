from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Locale:
    code: str
    native_name: str
    region: str
    strings: dict[str, str]


_EN = {
    "help": "HELP", "memory": "MEMORY", "context": "CONTEXT",
    "sessions": "SESSIONS", "settings": "SETTINGS + MODELS", "language": "LANGUAGE",
    "activity": "ACTIVITY STREAM", "agents": "AGENT TREE", "session_info": "SESSION INFO",
    "workspace": "CONVERSATION & WORKSPACE", "back": "← Back", "search": "Search",
    "new": "New", "switch": "Switch", "rename": "Rename", "close": "Close",
    "select_language": "SELECT LANGUAGE — Enter applies immediately",
}


def _locale(code: str, native: str, region: str, **translations: str) -> Locale:
    return Locale(code, native, region, {**_EN, **translations})


LOCALES = {
    "en": _locale("en", "English", "Global"),
    "fr": _locale("fr", "Français", "Africa / Global", help="AIDE", memory="MÉMOIRE", context="CONTEXTE", sessions="SESSIONS", settings="PARAMÈTRES + MODÈLES", language="LANGUE", activity="ACTIVITÉ", agents="ARBRE DES AGENTS", session_info="INFOS SESSION", workspace="CONVERSATION ET ESPACE", back="← Retour", search="Rechercher", new="Nouvelle", switch="Changer", rename="Renommer", close="Fermer", select_language="CHOISIR LA LANGUE — Entrée applique immédiatement"),
    "sw": _locale("sw", "Kiswahili", "East Africa", help="MSAADA", memory="KUMBUKUMBU", context="MUKTADHA", sessions="VIPINDI", settings="MIPANGILIO + MIUNDO", language="LUGHA", activity="MTIRIRIKO WA SHUGHULI", agents="MTI WA MAWAKALA", session_info="TAARIFA ZA KIPINDI", workspace="MAZUNGUMZO NA ENEO LA KAZI", back="← Rudi", search="Tafuta", new="Mpya", switch="Badili", rename="Badili jina", close="Funga", select_language="CHAGUA LUGHA — Enter inatumia mara moja"),
    "am": _locale("am", "አማርኛ", "Ethiopia", help="እገዛ", memory="ማህደረ ትውስታ", context="አውድ", sessions="ክፍለ ጊዜዎች", settings="ቅንብሮች + ሞዴሎች", language="ቋንቋ", activity="የእንቅስቃሴ ፍሰት", agents="የወኪሎች ዛፍ", session_info="የክፍለ ጊዜ መረጃ", workspace="ውይይት እና የስራ ቦታ", back="← ተመለስ", search="ፈልግ", new="አዲስ", switch="ቀይር", rename="እንደገና ሰይም", close="ዝጋ", select_language="ቋንቋ ይምረጡ — Enter ወዲያውኑ ይተገበራል"),
    "ha": _locale("ha", "Hausa", "West Africa", help="TAIMAKO", memory="ƘWAƘWALWA", context="MAHALLI", sessions="ZAMA", settings="SAITUNA + MISALAI", language="HARSHE", activity="AYYUKA", agents="BISHIYAR WAKILAI", session_info="BAYANIN ZAMA", workspace="TATTAUNAWA DA WURIN AIKI", back="← Koma", search="Bincika", new="Sabo", switch="Canja", rename="Sake suna", close="Rufe", select_language="ZAƁI HARSHE — Enter yana aiki nan take"),
    "yo": _locale("yo", "Yorùbá", "West Africa", help="ÌRÀNLỌ́WỌ́", memory="ÌRÁNTÍ", context="ÀYÍKÁ", sessions="ÀWỌN ÌPÀDÉ", settings="ÈTÒ + ÀWỌN ÀWÒRÁN", language="ÈDÈ", activity="ÌṢẸ́", agents="IGI AṢOJÚ", session_info="ÀLÀYÉ ÌPÀDÉ", workspace="ÌJÍRÒRÒ ATI ÀAYÈ IṢẸ́", back="← Padà", search="Wá", new="Tuntun", switch="Yípadà", rename="Tún lorúkọ", close="Pa", select_language="YAN ÈDÈ — Enter máa lò ó lẹ́sẹ̀kẹsẹ̀"),
    "zu": _locale("zu", "isiZulu", "Southern Africa", help="USIZO", memory="INKUMBULO", context="UMONGO", sessions="AMASESHINI", settings="IZILUNGISELELO + AMAMODELI", language="ULIMI", activity="UMSEBENZI", agents="ISIHLAHLA SAMA-EJENTI", session_info="ULWAZI LWESESHINI", workspace="INGXOXO NENDAWO YOKUSEBENZA", back="← Emuva", search="Sesha", new="Okusha", switch="Shintsha", rename="Qamba kabusha", close="Vala", select_language="KHETHA ULIMI — Enter isebenza ngokushesha"),
    "ar": _locale("ar", "العربية", "North Africa / West Asia", help="مساعدة", memory="الذاكرة", context="السياق", sessions="الجلسات", settings="الإعدادات + النماذج", language="اللغة", activity="سجل النشاط", agents="شجرة الوكلاء", session_info="معلومات الجلسة", workspace="المحادثة ومساحة العمل", back="رجوع ←", search="بحث", new="جديدة", switch="تبديل", rename="إعادة تسمية", close="إغلاق", select_language="اختر اللغة — Enter يطبق فورًا"),
    "hi": _locale("hi", "हिन्दी", "South Asia", help="सहायता", memory="स्मृति", context="संदर्भ", sessions="सत्र", settings="सेटिंग्स + मॉडल", language="भाषा", activity="गतिविधि", agents="एजेंट वृक्ष", session_info="सत्र जानकारी", workspace="वार्तालाप और कार्यक्षेत्र", back="← वापस", search="खोजें", new="नया", switch="बदलें", rename="नाम बदलें", close="बंद करें", select_language="भाषा चुनें — Enter तुरंत लागू करता है"),
    "bn": _locale("bn", "বাংলা", "South Asia", help="সহায়তা", memory="স্মৃতি", context="প্রসঙ্গ", sessions="সেশন", settings="সেটিংস + মডেল", language="ভাষা", activity="কার্যক্রম", agents="এজেন্ট ট্রি", session_info="সেশন তথ্য", workspace="কথোপকথন ও কর্মক্ষেত্র", back="← ফিরে যান", search="খুঁজুন", new="নতুন", switch="বদলান", rename="নাম বদলান", close="বন্ধ", select_language="ভাষা নির্বাচন করুন — Enter সঙ্গে সঙ্গে প্রয়োগ করে"),
    "ur": _locale("ur", "اردو", "South Asia", help="مدد", memory="یادداشت", context="سیاق", sessions="سیشن", settings="ترتیبات + ماڈلز", language="زبان", activity="سرگرمی", agents="ایجنٹ درخت", session_info="سیشن معلومات", workspace="گفتگو اور کام کی جگہ", back="واپس ←", search="تلاش", new="نیا", switch="تبدیل", rename="نام بدلیں", close="بند", select_language="زبان منتخب کریں — Enter فوراً لاگو کرتا ہے"),
    "zh": _locale("zh", "中文", "East Asia", help="帮助", memory="记忆", context="上下文", sessions="会话", settings="设置 + 模型", language="语言", activity="活动流", agents="智能体树", session_info="会话信息", workspace="对话与工作区", back="← 返回", search="搜索", new="新建", switch="切换", rename="重命名", close="关闭", select_language="选择语言 — 按 Enter 立即应用"),
    "ja": _locale("ja", "日本語", "East Asia", help="ヘルプ", memory="メモリ", context="コンテキスト", sessions="セッション", settings="設定 + モデル", language="言語", activity="アクティビティ", agents="エージェントツリー", session_info="セッション情報", workspace="会話とワークスペース", back="← 戻る", search="検索", new="新規", switch="切替", rename="名前変更", close="閉じる", select_language="言語を選択 — Enterですぐ適用"),
    "ko": _locale("ko", "한국어", "East Asia", help="도움말", memory="메모리", context="컨텍스트", sessions="세션", settings="설정 + 모델", language="언어", activity="활동", agents="에이전트 트리", session_info="세션 정보", workspace="대화 및 작업공간", back="← 뒤로", search="검색", new="새로 만들기", switch="전환", rename="이름 변경", close="닫기", select_language="언어 선택 — Enter로 즉시 적용"),
    "id": _locale("id", "Bahasa Indonesia", "Southeast Asia", help="BANTUAN", memory="MEMORI", context="KONTEKS", sessions="SESI", settings="PENGATURAN + MODEL", language="BAHASA", activity="AKTIVITAS", agents="POHON AGEN", session_info="INFO SESI", workspace="PERCAKAPAN & RUANG KERJA", back="← Kembali", search="Cari", new="Baru", switch="Pindah", rename="Ubah nama", close="Tutup", select_language="PILIH BAHASA — Enter langsung menerapkan"),
}


def translate(code: str, key: str) -> str:
    return LOCALES.get(code, LOCALES["en"]).strings.get(key, _EN.get(key, key))
