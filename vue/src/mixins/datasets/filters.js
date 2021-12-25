export default {
  filters: {
    formatDate: value => {
      const date = new Date(value);
      return date.toLocaleString(['ru-RU'], {
        hour: '2-digit',
        minute: '2-digit',
        month: '2-digit',
        day: '2-digit',
        year: '2-digit',
      });
    },
  },
};
