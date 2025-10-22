# Baby Care Assistant - Modern Frontend

A modern, responsive chat interface built with Next.js 15, TypeScript, and Tailwind CSS.

## 🎨 Features

### Modern Chat Interface
- **ChatGPT-style Design**: Clean, professional chat bubbles with avatars
- **Real-time Typing Indicators**: Shows "Thinking..." while AI processes
- **Auto-resizing Input**: Textarea grows with content
- **Smooth Animations**: Hover effects and transitions
- **Responsive Design**: Works on desktop, tablet, and mobile

### User Experience
- **Welcome Screen**: Beautiful landing page with example questions
- **Message Timestamps**: Shows when messages were sent
- **Document Sources**: Displays retrieved documents with sources
- **Keyboard Shortcuts**: Enter to send, Shift+Enter for new line
- **New Chat Button**: Easy conversation reset

### Technical Features
- **Next.js 15**: Latest features including React 19 support
- **TypeScript**: Full type safety with strict configuration
- **Component Architecture**: Modular, reusable components
- **Auto-scroll**: Automatically scrolls to new messages
- **Error Handling**: Graceful error states
- **Loading States**: Visual feedback during API calls
- **Performance**: React Compiler, Partial Prerendering, optimized bundles
- **Security**: Built-in security headers and XSS protection

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## 📁 Project Structure

```
frontend/
├── app/
│   ├── page.tsx          # Main chat page
│   ├── layout.tsx        # App layout
│   └── globals.css       # Global styles
├── components/
│   ├── MessageBubble.tsx # Individual message component
│   ├── WelcomeScreen.tsx # Landing page component
│   └── ChatInput.tsx     # Input component
├── package.json          # Dependencies
└── tailwind.config.js    # Tailwind configuration
```

## 🎨 Design System

### Colors
- **Primary**: Blue (#3B82F6)
- **Secondary**: Purple gradient
- **Background**: Light gray (#F9FAFB)
- **Text**: Dark gray (#111827)

### Components
- **Message Bubbles**: Rounded corners, shadows, hover effects
- **Avatars**: Circular with icons (User/Bot)
- **Buttons**: Rounded, hover states, disabled states
- **Input**: Auto-resizing textarea with send button

### Typography
- **Font**: Inter (system font stack)
- **Sizes**: Responsive text sizing
- **Weights**: Regular, medium, semibold

## 🔧 Customization

### Styling
Edit `app/globals.css` for custom styles:
```css
/* Custom scrollbar */
::-webkit-scrollbar {
  width: 6px;
}

/* Message bubble shadows */
.message-bubble {
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}
```

### Components
Each component is modular and can be customized:
- `MessageBubble.tsx`: Message display logic
- `WelcomeScreen.tsx`: Landing page content
- `ChatInput.tsx`: Input handling

### API Integration
Update the API base URL in `app/page.tsx`:
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
```

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 640px (single column)
- **Tablet**: 640px - 1024px (responsive grid)
- **Desktop**: > 1024px (sidebar + main chat)

### Mobile Features
- **Touch-friendly**: Large tap targets
- **Swipe gestures**: Natural mobile interactions
- **Keyboard support**: Mobile keyboard optimization

## 🚀 Deployment

### Vercel (Recommended)
```bash
# Deploy to Vercel
npx vercel

# Set environment variables
NEXT_PUBLIC_API_URL=https://your-api-url.com
```

### Netlify
```bash
# Build the project
npm run build

# Deploy to Netlify
# Upload the 'out' folder
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🔧 Development

### Available Scripts
- `npm run dev`: Start development server
- `npm run build`: Build for production
- `npm run start`: Start production server
- `npm run lint`: Run ESLint

### Environment Variables
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🎯 Performance

### Optimizations
- **Code Splitting**: Automatic with Next.js
- **Image Optimization**: Next.js Image component
- **Bundle Analysis**: Built-in bundle analyzer
- **Lazy Loading**: Components loaded on demand

### Best Practices
- **TypeScript**: Full type safety
- **Component Reusability**: Modular architecture
- **Accessibility**: ARIA labels and keyboard navigation
- **SEO**: Meta tags and structured data

## 🎨 UI/UX Features

### Chat Experience
- **Message Threading**: Clear conversation flow
- **Typing Indicators**: Real-time feedback
- **Message Status**: Sent, delivered, read states
- **Rich Content**: Support for links, formatting

### Accessibility
- **Keyboard Navigation**: Full keyboard support
- **Screen Readers**: ARIA labels and descriptions
- **Color Contrast**: WCAG compliant colors
- **Focus Management**: Clear focus indicators

## 🔮 Future Enhancements

### Planned Features
- **Dark Mode**: Toggle between light/dark themes
- **Message Search**: Search through conversation history
- **File Uploads**: Support for image/document uploads
- **Voice Input**: Speech-to-text integration
- **Message Reactions**: Emoji reactions to messages

### Advanced Features
- **Real-time Updates**: WebSocket integration
- **Message Threading**: Reply to specific messages
- **Custom Themes**: User-customizable appearance
- **Offline Support**: Service worker integration
- **Push Notifications**: Browser notifications

## 📊 Analytics

### Built-in Metrics
- **Performance**: Core Web Vitals
- **User Engagement**: Message counts, session duration
- **Error Tracking**: JavaScript errors, API failures
- **Usage Patterns**: Most common questions, peak times

## 🛡️ Security

### Best Practices
- **Input Sanitization**: XSS prevention
- **CSRF Protection**: Cross-site request forgery prevention
- **Content Security Policy**: XSS mitigation
- **HTTPS Only**: Secure connections required

## 📈 Monitoring

### Health Checks
- **API Connectivity**: Backend connection status
- **Error Rates**: Failed request tracking
- **Performance**: Response time monitoring
- **User Experience**: Real user monitoring

This modern frontend provides a professional, user-friendly interface for the Baby Care Assistant, combining the best practices of modern web development with an intuitive chat experience.
