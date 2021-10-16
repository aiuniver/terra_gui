<template>
  <div>
    <t-button :disabled="!selected" @click.native="handleClick">
      <span v-if="selected">Выбрать датасет</span>
      <span v-else>{{ btnText }}</span>
    </t-button>
  </div>
</template>

<script>
export default {
  name: 'DatasetButton',
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
    btnText() {
      const name = this.$store.getters['projects/getProject']?.dataset?.name;
      if (name) return 'Выбран: ' + name;
      return 'Выберите датасет';
    },
  },
  methods: {
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
              this.$store.dispatch('messages/setMessage', {
                message: `Датасет «${data?.data?.dataset?.name || ''}» выбран`,
              });
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
      }, 1000);
    },
    async isTraining() {
      return await this.$store.dispatch('dialogs/trining', { ctx: this, page: 'датасета' });
    },
    async handleClick() {
      const dataset = this.selected;
      const isTrain = await this.isTraining();
      if (isTrain) {
        const { success, data } = await this.$store.dispatch('datasets/validateDatasetOrModel', {
          dataset,
        });
        if (success && data) {
          const answer = await this.$store.dispatch('dialogs/confirm', { ctx: this, content: data });
          if (answer == 'confirm') await this.onChoice({ ...dataset, reset_model: true });
        } else {
          await this.onChoice({ ...dataset, reset_model: false });
        }
      }
    },
    async onChoice({ alias, group, name, reset_model } = {}) {
      this.$store.dispatch('settings/setOverlay', true);
      const { success: successChoice } = await this.$store.dispatch('datasets/choice', { alias, group, reset_model });

      if (successChoice) {
        this.$store.dispatch('messages/setMessage', { message: `Загружаю датасет «${name}»` });
        this.createInterval();
      }
    },
  },
};
</script>
