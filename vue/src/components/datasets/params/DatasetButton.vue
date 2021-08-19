<template>
  <div>
    <t-button :disabled="!selected" :loading="loading" @click.native="click">Выбрать датасет</t-button>
  </div>
</template>

<script>
export default {
  name: 'DatasetButton',
  data: () => ({
    loading: false
  }),
  computed: {
    selected() {
      return this.$store.getters['datasets/getSelected'];
    },
    selectedIndex(){
      return this.$store.getters['datasets/getSelectedIndex'];
    }
  },
  methods: {
    createInterval() {
      this.interval = setTimeout(async () => {
        const { data } = await this.$store.dispatch('datasets/choiceProgress', {});
        const { finished, message, percent, data: dataset } = data;
        if (!data || finished) {
          this.$store.dispatch('messages/setProgressMessage', message);
          this.$store.dispatch('messages/setProgress', percent);
          this.loading = false
          if (data) {
            this.$store.dispatch(
              'messages/setMessage',
              { message: `Датасет «${dataset.alias}» выбран` },
              { root: true }
            );
            this.$store.dispatch('projects/setProject', { dataset }, { root: true });
            this.$store.dispatch('datasets/setLoaded', this.selectedIndex);
          }
        } else {
          this.$store.dispatch('messages/setProgress', percent);
          this.$store.dispatch('messages/setProgressMessage', message);
          this.createInterval();
        }
        console.log(data);
      }, 1000);
    },
    async click() {
      const { alias, group, name } = this.selected;
      this.$store.dispatch('messages/setMessage', {
        message: `Выбран датасет «${name}»`,
      });
      const { data } = await this.$store.dispatch('datasets/choice', { alias, group });
      if (data) {
        this.loading = true
        this.createInterval();
      }
    },
  },
};
</script>
