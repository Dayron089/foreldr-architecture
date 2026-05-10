# Screenshots

Сюда положи финальные скрины **web-PWA FORELDR**. Имена файлов фиксированные — главный README ссылается на них напрямую через `screenshots/<имя>.png`.

## Что и под каким именем

### Desktop / large screens — приоритет 1

- `01-web-chat.png` — экран диалога с персонажем (видны bubbles, поле ввода, аватар, sidebar со списком чатов). **Это hero-скрин** — самый видный в README.
- `02-web-chat-list.png` — список чатов с unread-counters и last-message preview
- `03-web-discover.png` — discovery / выбор персонажей (карточки)
- `04-web-character-profile.png` — профиль персонажа (фото, описание, scenarios)
- `05-web-stories.png` — лента историй персонажей (Instagram-like)

### Mobile browser (responsive PWA) — приоритет 2

PWA на телефоне в браузере (iOS Safari или Chrome Android), портретный режим:

- `06-mobile-chat.png` — мобильный экран чата
- `07-mobile-list.png` — мобильный список чатов
- `08-mobile-discover.png` — мобильный discovery

### Admin / Observability — приоритет 3

Это сильно прокачивает портфолио — показывает что система не только работает, но и **наблюдаема**.

- `09-admin-cost-dashboard.png` — дашборд стоимости с разбивкой по моделям, cache hit rate, total spend
- `10-admin-moderation.png` — панель модерации (нарушения, репорты, real-time feed)
- `11-cron-jobs.png` — статус cron-задач (BER sync, contact initiator, story scheduler)

## Рекомендации по подготовке

### Размер и формат
- **Desktop:** разрешение 1440×900 или выше, ratio близкий к 16:10. Браузер на full-screen, без вкладок и URL-бара (или закрашены) для чистоты.
- **Mobile:** оригинальное разрешение портретного экрана телефона (например 390×844 для iPhone, 412×915 для Android-mid).
- **Формат:** **PNG** — без артефактов на тексте. JPG используй только если итоговый файл получается > 1 MB.
- **Размер файла:** оптимизируй через [tinypng.com](https://tinypng.com) или `pngquant` — < 500 KB на скрин для быстрой загрузки на GitHub.

### Содержимое
- Никаких реальных юзеров и переписок. Заведи демо-аккаунт с тестовыми персонажами и "разогрей" его на 5-10 диалогов с разнообразным контентом.
- В чатах должны быть видны разные элементы UI: bubbles разных типов, timestamps, индикатор "печатает", emoji-реакции, прикреплённое фото от персонажа (если есть в скрине).
- Диалог должен выглядеть **естественно** — не "Hello", "Hi", а настоящий обмен на 4-5 реплик с конкретикой.

### Чувствительные данные — проверь перед коммитом
- Удали email, реальные user IDs из URL'ов
- Замазь admin email на дашборде
- Скрой JWT токены если светятся в DevTools / network tab
- В admin-панели — анонимизируй имена реальных юзеров (например через Cmd+F → Replace в скрине)

### Инструменты для скринов
- **macOS:** `Cmd+Shift+4` → область, `Cmd+Shift+4` затем Space → конкретное окно. Сохраняется на Desktop.
- **CleanShot X / Shottr** — лучше встроенного инструмента, есть аннотации, размытие, шаблоны.
- **Browser DevTools** для мобильных скринов: F12 → Toggle device toolbar → выбери iPhone 14 Pro / Pixel 7 → screenshot из Capture menu.

## Опционально — прокачка

Если хочешь поднять уровень:

1. **Mockup-обёртки.**
   - [mockuphone.com](https://mockuphone.com) — рамки телефонов / ноутбуков, бесплатно
   - [shots.so](https://shots.so) — стильные браузерные mockup'ы с тенями
   - [screenshot.rocks](https://screenshot.rocks) — chrome window mockup
2. **Hero-коллаж.** Объединить 3-4 топ-скрина в один `00-hero.png` для самого верха README — это первое что видят рекрутеры. Figma / Photopea (бесплатно).
3. **Annotated screenshots.** Стрелочки и подписи на admin-дашборде типа "cache hit rate 96%" или "cost per message $0.0019" — мгновенно понятно куда смотреть. CleanShot X умеет это из коробки.
4. **GIF / video demo.** Запись 5-10 секундного флоу диалога с персонажем (`07-demo.gif`) — оживляет README. Инструменты: [Kap](https://getkap.co), CleanShot.

## После того как положишь скрины

```bash
cd /Users/dimapelih/Desktop/foreldr-architecture
git add screenshots/
git commit -m "Add product screenshots"
git push
```
