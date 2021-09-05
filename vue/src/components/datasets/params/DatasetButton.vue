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

            this.$store.dispatch('messages/setProgress', 0);
            this.$store.dispatch('messages/setProgressMessage', '');
            await this.$store.dispatch('projects/get');
            const { data: dataset } = data;
            this.$store.dispatch(
              'messages/setMessage',
              { message: `Датасет «${dataset.name}» выбран` },
              { root: true }
            );
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
      const { success } = await this.$store.dispatch('datasets/choice', { alias, group });
      this.$store.dispatch('messages/setMessage', { message: `Загружаю датасет «${name}»` });
      if (success) {
        this.createInterval();
      }
    },
  },
};
</script>
