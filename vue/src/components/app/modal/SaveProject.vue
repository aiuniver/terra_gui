<template>
  <at-modal v-model="dialog" width="400" :maskClosable="false" :showClose="false">
    <div slot="header">
      <span>Сохранить проект</span>
    </div>
    <div class="inner form-inline-label">
      <t-field label="Название проекта">
        <d-input-text v-model="name" type="text" :disabled="loading" />
      </t-field>
      <t-field label="Перезаписать">
        <d-checkbox v-model="overwrite" type="checkbox" :disabled="loading" />
      </t-field>
    </div>
    <template slot="footer">
      <t-button @click="save({ name, overwrite })" :loading="loading">Сохранить</t-button>
      <t-button @click="dialog = false" cancel :disabled="loading">Отменить</t-button>
    </template>
  </at-modal>
</template>

<script>
import DCheckbox from '../../../../../newue/src/components/global/forms/DCheckbox.vue';
export default {
  components: { DCheckbox },
  name: 'modal-save-project',
  props: {
    value: Boolean,
  },
  data: () => ({
    name: '',
    overwrite: false,
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
        this.$emit('message', { message: `Сохранения проекта «${data.name}»` });
        const res = await this.$store.dispatch('projects/saveProject', data);
        if (res && !res.error) {
          this.$emit('message', { message: `Проект «${data.name}» сохранен` });
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
        this.name = this.$store.getters['projects/getProject'].name;
      }
    },
  },
};
</script>