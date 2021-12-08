export default {
  filters: {
    formatDate: value => {
      const date = new Date(value);
      return date.toLocaleString(['ru-RU'], {
        month: 'short',
        day: '2-digit',
        // year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    },
  },
};
