<template>
  <at-modal v-model="dialog" width="500" showClose>
    <div slot="header" style="text-align: center">
      <span>Сохранить модель</span>
    </div>
    <div class="model modal-save-model">
      <div v-if="image" class="model__image">
        <img alt="" width="auto" height="400" :src="image || ''" />
      </div>
      <div class="model__config">
        <t-input
          :value="name"
          :label="'Название проекта'"
          :type="'text'"
          :parse="'parse'"
          :name="'name'"
          :key="'name-key'"
          @change="name = $event.value"
        />
        <t-checkbox
          inline
          :value="overwrite"
          :label="'Перезаписать'"
          type="checkbox"
          :parse="'test'"
          :name="'overwrite'"
          :key="'overwrite'"
          @change="overwrite = $event.value"
        />
      </div>
      <Loading v-if="!image" />
    </div>
    <template slot="footer">
      <button @click="save">Сохранить</button>
      <button class="at-btn at-btn--default" @click="close">
        <span class="at-btn__text">Отменить</span>
      </button>
    </template>
  </at-modal>
</template>

<script>
import Loading from '../../forms/Loading.vue';

export default {
  name: 'ModalSaveModel',
  components: {
    Loading,
  },
  props: {
    value: Boolean,
    image: String,
  },
  data: () => ({
    name: '',
    overwrite: false,
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
    change(e) {
      console.log(e);
    },
    async save() {
      const res = await this.$store.dispatch('modeling/createModel', {
        name: this.name,
        preview: this.image.slice(22),
        overwrite: this.overwrite,
      });
      this.dialog = false;
      console.log(this.name);
      console.log(this.overwrite);
      console.log(res);
      // this.$emit('input', false);
    },

    close() {
      this.dialog = false;
    },
  },
  watch: {
    dialog: {
      handler(value) {
        if (value) {
          console.log(this.refs);
        }
      },
    },
  },
};
</script>

<style lang="scss" scoped>
.scroll-area {
  height: 350px;
}
.modal-save-model {
  flex-direction: column;

  [type='checkbox'] {
    margin-top: 10px;
  }
}
.model {
  display: flex;
  justify-content: center;

  &__image {
    height: 400px;
    img {
      width: 100%;
    }
  }
}
</style>
