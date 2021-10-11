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
    async message() {
      await this.$store.dispatch('messages/setModel', {
        context: this,
        content: 'Для выбора датасета необходимо сбросить/остановить обучение',
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
      }, 1000);
    },
    async click() {
      if (this.isNoTrain) {
        const { alias, group, name} = this.selected;
       
        const { success: successValidate, data } = await this.$store.dispatch('datasets/validateDatasetOrModel', {
          dataset: { alias, group },
        });

        this.$store.dispatch('messages/setMessage', { message: `Загружаю датасет «${name}»` });

        if (successValidate && !data) {
          this.$Modal.confirm({
            title: 'Внимание!',
            content:
              'Несоответствие количества входных и выходных слоев датасета и редактируемой модели. Хотите сбросить модель?',
            width: 300,
            callback:  (action) => {
              if (action == 'confirm') {
                this.createInterval();
                
              }
            },
          });
           
        } else {
          this.$store.dispatch('settings/setOverlay', true);
          const { success: successChoice } = await this.$store.dispatch('datasets/choice', { alias, group });
          if (successChoice) {
            this.createInterval();
          }
        }
      } else {
        this.message();
      }

    },
  },
};
</script>
