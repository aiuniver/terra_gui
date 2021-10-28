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
  data: () => ({
    mixinUpdateDate: {},
    mixinFilter: {
      Image: ['folder', 'table'],
      Text: ['folder', 'table'],
      Audio: ['folder', 'table'],
      Video: ['folder', 'table'],
      Classification: ['table'],
      Segmentation: ['folder', 'table'],
      Regression: ['table'],
      Timeseries: ['table'],
      ObjectDetection: ['folder', 'table'],
      Dataframe: ['table'],
    }
  }),
  methods: {
    mixinUpdate({ id, value }) {
      if (value.length) {
        const files = this.$store.getters['datasets/getFilesSource'] || []
        const { extra } = files.find(item => item.path === value[0].value);
        if (extra) {
          for (let key in extra) {
            this.mixinChange({ id, name: key, value: extra[key] });
          }
          this.mixinUpdateDate = {};
          this.$nextTick(() => {
            this.mixinUpdateDate = extra;
          });
        }
      }
    },
    mixinCheck(selected, id) {
      this.mixinFiles = this.mixinFiles.map(file => {
        if (selected.find(item => item.value === file.value)) {
          file.id = id;
        } else {
          file.id = (file.id === id) ? 0 : file.id;
        }
        return file;
      });
      const value = selected.map(item => item.value)
      this.mixinChange({ id, name: 'sources_paths', value })
    },
    mixinRemove(id) {
      this.$store.dispatch('tables/setSaveCols', {id, value: []});
      this.mixinFiles = this.mixinFiles.map(item => {
        item.id = (item.id === id) ? 0 : item.id;
        return item;
      });
    },
    mixinChange(obj) {
      if (obj.name === 'type') {
        console.log(obj)
        this.mixinRemove(obj.id)
      }

      this.$store.dispatch('datasets/updateInputData', obj)
    }
  },
};
