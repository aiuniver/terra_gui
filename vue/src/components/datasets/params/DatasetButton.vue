<template>
  <div>
    <t-button :disabled="!selected" @click.native="click">Выбрать датасет</t-button>
  </div>
</template>

<script>
export default {
  name: 'DatasetButton',
  data: () => ({}),
  computed: {
    isNoTrain() {
      return this.$store.getters['trainings/getStatus'] === 'no_train';
    },
    selected() {
      return this.$store.getters['datasets/getSelected'];
    },
    selectedIndex() {
      return this.$store.getters['datasets/getSelectedIndex'];
    },
  },
  methods: {
    async message() {
      await this.$store.dispatch('messages/setModel', {
        context: this,
        content: 'Для выбора датасета остановите обучение',
      });
    },
    createInterval() {
      this.interval = setTimeout(async () => {
        const res = await this.$store.dispatch('datasets/choiceProgress', {});
        if (res) {
          const { data } = res;
          if (data) {
            const { finished, message, percent, error } = data;
            this.$store.dispatch('messages/setProgressMessage', message);
            this.$store.dispatch('messages/setProgress', percent);
            if (finished) {
              this.$store.dispatch('messages/setProgress', 0);
              this.$store.dispatch('messages/setProgressMessage', '');
              await this.$store.dispatch('projects/get');
              const { data: dataset } = data;
              this.$store.dispatch(
                'messages/setMessage',
                { message: `Датасет «${dataset.name}» выбран` },
                { root: true }
              );
              this.$store.dispatch('settings/setOverlay', false);
            } else {
              if (error) {
                this.$store.dispatch('messages/setMessage', { error });
                this.$store.dispatch('messages/setProgressMessage', '');
                this.$store.dispatch('messages/setProgress', 0);
                this.$store.dispatch('settings/setOverlay', false);
                return;
              }
              this.createInterval();
            }
          } else {
            this.$store.dispatch('settings/setOverlay', false);
          }
        } else {
          this.$store.dispatch('settings/setOverlay', false);
        }
        // console.log(data);
      }, 1000);
    },
    async click() {
      if (this.isNoTrain) {
        this.$store.dispatch('settings/setOverlay', true);
        const { alias, group, name } = this.selected;
        const { success } = await this.$store.dispatch('datasets/choice', { alias, group });
        this.$store.dispatch('messages/setMessage', { message: `Загружаю датасет «${name}»` });
        if (success) {
          this.createInterval();
        }
      } else {
        this.message();
      }
    },
  },
};
</script>
