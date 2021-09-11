<template>
  <at-modal v-model="dialog" width="500" showClose>
    <div slot="header" style="text-align: center">
      <span>Сохранить модель</span>
    </div>
    <div class="model modal-save-model">
      <div class="model__image">
        <Loading v-if="!image" />
        <img v-if="image" alt="" width="auto" height="auto" :src="image || ''" />
      </div>
      <div class="model__config">
        <t-input
          :value="name"
          :label="'Название модели'"
          :type="'text'"
          :parse="'parse'"
          :name="'name'"
          :key="'name-key'"
          :error="err"
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
    </div>
    <template slot="footer">
      <t-button @click.native="save">Сохранить</t-button>
      <t-button cancel @click.native="close">Отменить</t-button>
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
    err: '',
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
      this.err = null;

      const res = await this.$store.dispatch('modeling/createModel', {
        name: this.name,
        preview: this.image.slice(22),
        overwrite: this.overwrite,
      });

      console.log(res);

      if (res?.error && res?.error?.general) {
        await this.$store.dispatch(
          'messages/setMessage',
          { error: `Moдель '${this.name}' уже создана` },
          { root: true }
        );
        this.err = `Moдель '${this.name}' уже создана`;
      } else if (res?.error) {
        await this.$store.dispatch('messages/setMessage', { error: 'Поле не может быть пустым' }, { root: true });
        this.err = 'Поле не может быть пустым';
      }
      console.log(`this.err ${this.err}`);

      if (!this.err) {
        this.dialog = false;
        await this.$store.dispatch(
          'messages/setMessage',
          { message: `Moдель '${this.name}' сохранена` },
          { root: true }
        );
      }
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
    height: auto;
    width: 100%;
    margin-bottom: 20px;
    img {
      width: 100%;
    }
  }
}
</style>
