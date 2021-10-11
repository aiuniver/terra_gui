<template>
  <div>
    <t-button :disabled="!selected" @click.native="click">
      <span v-if="selected">Выбрать датасет</span>
      <span v-else>{{ btnText }}</span>
    </t-button>
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
    btnText() {
      const name = this.$store.getters['projects/getProject']?.dataset?.name;
      if (name) return 'Выбран: ' + name;
      return 'Выберите датасет';
    }
  },
  methods: {
    async message(content) {
      await this.$store.dispatch('messages/setModel', {
        context: this,
        content,
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
        if (success) {
          this.$store.dispatch('messages/setMessage', { message: `Загружаю датасет «${name}»` });
          this.createInterval();
        }else{
          this.$store.dispatch('settings/setOverlay', false);
          this.message('Валидация датасета/модели не прошла');
        }
      } else {
        this.message('Для выбора датасета необходимо сбросить/остановить обучение');
      }
    },
  },
};
</script>
