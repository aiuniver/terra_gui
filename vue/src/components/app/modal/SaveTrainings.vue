<template>
  <at-modal v-model="dialog" width="400" :maskClosable="false" :showClose="false">
    <div slot="header">
      <span>Сохранить обучение</span>
    </div>
    <div class="inner form-inline-label">
      <t-field label="Название">
        <d-input-text v-model="name" type="text" :disabled="loading" />
      </t-field>
      <t-field label="Перезаписать">
        <d-checkbox v-model="overwrite" type="checkbox" :disabled="loading" />
      </t-field>
    </div>
    <template slot="footer">
      <t-button :disabled="name === ''" :loading="loading" @click="save({ name, overwrite })">Сохранить</t-button>
      <t-button cancel :disabled="loading" @click="dialog = false">Отменить</t-button>
    </template>
  </at-modal>
</template>

<script>
export default {
  name: 'modal-save-project',
  props: {
    value: Boolean,
  },
  data: () => ({
    name: '',
    overwrite: true,
    loading: false,
    disabled: false,
  }),
  computed: {
    dialog: {
      set(value) {
        this.$emit('input', value);
      },
      get() {
        return this.value;
      },
    },
  },
  methods: {
    async save(data) {
      try {
        this.loading = true;
        // console.log(data)
        // this.$emit('message', { message: `Сохранения проекта «${data.name}»` });
        const res = await this.$store.dispatch('trainings/save', data);
        if (res && !res.error) {
          this.$emit('message', { message: `Проект «${data.name}» сохранен` });
          await this.$store.dispatch('projects/get', {});
          this.dialog = false;
          this.overwrite = false;
        } else {
          this.$emit('message', { error: res.error.general });
        }
        this.loading = false;
      } catch (error) {
        console.log(error);
        this.loading = false;
      }
    },
  },
  watch: {
    dialog(value) {
      if (value) {
        const name = this.$store.getters['projects/getProject'].training.name;
        this.name = name === '__current' ? '' : name;
      }
    },
  },
};
</script>