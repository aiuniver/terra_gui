export default {
  computed: {
    mixinFiles: {
      set(value) {
        this.$store.dispatch('datasets/setFilesDrop', value);
      },
      get() {
        return this.$store.getters['datasets/getFilesDrop'];
      },
    },
  },
  methods: {
    mixinCheck(selected, id) {
      this.mixinFiles = this.mixinFiles.map(file => {
        if (selected.find(item => item.value === file.value)) {
          file.id = id;
        } else {
          file.id = (file.id === id) ? 0 : file.id;
        }
        return file;
      });
    },
    mixinRemove(id) {
      this.mixinFiles = this.mixinFiles.map(item => {
        item.id = (item.id === id) ? 0 : item.id;
        return item;
      });
    },
  },
};
