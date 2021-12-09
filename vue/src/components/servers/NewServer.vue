<template>
  <div class="new-server">
    <p class="new-server__header">Добавление сервера демо-панели</p>
    <form @submit.prevent class="new-server__form" @submit="$emit('addserver')">
      <t-field label="Доменное имя">
        <DInputText placeholder="" />
      </t-field>
      <t-field label="IP адрес">
        <VueIP :ip="ip" :onChange="change" />
      </t-field>
      <t-field label="Имя пользователя">
        <DInputText placeholder="" />
      </t-field>
      <div class="new-server__ports">
        <t-field label="SSH порт">
          <DInputNumber placeholder="" />
        </t-field>
        <t-field label="HTTP порт">
          <DInputNumber placeholder="" />
        </t-field>
        <t-field label="HTTPS порт">
          <DInputNumber placeholder="" />
        </t-field>
      </div>
      <button class="new-server__btn" :disabled="!ipValid">Добавить</button>
    </form>
    <div class="new-server__error">
      <p class="new-server__error--header">Ошибка добавления сервера демо-панели</p>
      <p class="new-server__error--info">
        Неверно указан IP адрес. Отредактируйте поле и снова нажмите кнопку Добавить
      </p>
    </div>
  </div>
</template>

<script>
import VueIP from './VueIp.vue';

export default {
  name: 'NewServer',
  data: () => ({
    ip: '255.255.255.255',
    ipValid: null,
  }),
  components: {
    VueIP,
    DInputNumber: () => import('@/components/global/design/forms/components/DInputNumber'),
    DInputText: () => import('@/components/global/design/forms/components/DInputText'),
  },
  methods: {
    change(ip, x, valid) {
      this.ip = ip;
      this.ipValid = valid;
    },
  },
};
</script>

<style lang="scss" scoped>
.new-server {
  height: 100%;
  position: relative;
  &__header {
    padding: 15px 20px;
    font-size: 14px;
    font-weight: 600;
    color: #fff;
    border-bottom: #0e1621 1px solid;
  }
  &__form {
    padding: 30px 20px;
  }
  &__ports {
    display: flex;
    gap: 10px;
    > * {
      flex: 1 0 90px;
    }
  }
  &__btn {
    margin-top: 65px;
  }
  &__error {
    position: absolute;
    right: 0;
    bottom: 100px;
    background: #242f3d;
    border-radius: 4px;
    font-size: 14px;
    max-width: 330px;
    padding: 15px;
    &--header {
      color: #f2f5fa;
      margin-bottom: 10px;
      font-size: inherit;
    }
    &--info {
      color: #a7bed3;
      font-size: inherit;
    }
  }
}
</style>