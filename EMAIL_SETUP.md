# Email Setup Guide

To receive daily digests via email, you need to configure SMTP settings in your `.env` file.

## Gmail Setup (Recommended)

1. **Enable 2-Factor Authentication** on your Gmail account (if not already enabled)

2. **Generate an App Password**:
   - Go to https://myaccount.google.com/apppasswords
   - Select "Mail" and "Other (Custom name)"
   - Enter "AI News Agent" as the name
   - Click "Generate"
   - Copy the 16-character password (you'll use this as `SMTP_PASSWORD`)

3. **Add to `.env` file**:
   ```bash
   # Email Configuration
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-16-char-app-password
   EMAIL_FROM=your-email@gmail.com
   EMAIL_TO=robert.burden@gmail.com
   ```

## Other Email Providers

### Outlook/Hotmail
```bash
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587
SMTP_USERNAME=your-email@outlook.com
SMTP_PASSWORD=your-password
```

### Custom SMTP
```bash
SMTP_SERVER=your-smtp-server.com
SMTP_PORT=587
SMTP_USERNAME=your-username
SMTP_PASSWORD=your-password
EMAIL_FROM=your-email@domain.com
EMAIL_TO=robert.burden@gmail.com
```

## Testing

Run the environment check:
```bash
python check_env.py
```

This will verify your email configuration is set up correctly.

## Security Note

Never commit your `.env` file to git. It should already be in `.gitignore`.
