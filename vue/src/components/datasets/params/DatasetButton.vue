<template>
  <div>
    <t-button :disabled="!selected" :loading="loading" @click.native="click">Выбрать датасет</t-button>
  </div>
</template>

<script>
export default {
  name: 'DatasetButton',
  data: () => ({
    loading: false,
  }),
  computed: {
    selected() {
      return this.$store.getters['datasets/getSelected'];
    },
    selectedIndex() {
      return this.$store.getters['datasets/getSelectedIndex'];
    },
  },
  methods: {
    createInterval() {
      this.interval = setTimeout(async () => {
        const { data } = await this.$store.dispatch('datasets/choiceProgress', {});
        if (data) {
          const { finished, message, percent, error } = data;
          this.$store.dispatch('messages/setProgressMessage', message);
          this.$store.dispatch('messages/setProgress', percent);
          if (finished) {
            this.loading = false;
            const { data: dataset } = data;
            this.$store.dispatch(
              'messages/setMessage',
              { message: `Датасет «${dataset.alias}» выбран` },
              { root: true }
            );
            this.$store.dispatch('messages/setProgress', 0);
            this.$store.dispatch('messages/setProgressMessage', '');
            this.$store.dispatch('projects/get');
          } else {
            if (error) {
              this.$store.dispatch('messages/setMessage', { error });
              this.$store.dispatch('messages/setProgressMessage', '');
              this.$store.dispatch('messages/setProgress', 0);
              this.loading = false;
              return;
            }
            this.createInterval();
          }
        }
        // console.log(data);
      }, 1000);
    },
    async click() {
      if (this.loading) return;
      this.loading = true;
      const { alias, group, name } = this.selected;
      this.$store.dispatch('messages/setMessage', { message: `Выбран датасет «${name}»` });
      const { success } = await this.$store.dispatch('datasets/choice', { alias, group });
      if (success) {
        // this.$store.dispatch('messages/setMessage', { message: `Загружаю датасет «${name}»`,});
        this.createInterval();
      }
    },
  },
};
</script>
