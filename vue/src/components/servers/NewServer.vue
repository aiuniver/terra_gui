<template>
  <div class="new-server">
    <p class="new-server__header">Добавление сервера демо-панели</p>
    <form class="new-server__form" @submit.prevent="addServer">
      <div class="new-server__form--overlay" v-show="loading"></div>
      <t-field label="Доменное имя">
        <t-input-new placeholder="" v-model="domain_name" />
      </t-field>
      <t-field label="IP адрес">
        <VueIP :ip="ip_address" :onChange="change" />
      </t-field>
      <t-field label="Имя пользователя">
        <t-input-new placeholder="" v-model="user" />
      </t-field>
      <div class="new-server__ports">
        <t-field label="SSH порт">
          <d-input-number placeholder="" v-model="port_ssh" />
        </t-field>
        <t-field label="HTTP порт">
          <d-input-number placeholder="" v-model="port_http" />
        </t-field>
        <t-field label="HTTPS порт">
          <d-input-number placeholder="" v-model="port_https" />
        </t-field>
      </div>
      <t-button class="new-server__btn" :disabled="!validForm || loading">Добавить</t-button>
    </form>
    <LoadSpiner v-show="loading" text="" />
    <!-- <div class="new-server__error">
      <p class="new-server__error--header">Ошибка добавления сервера демо-панели</p>
      <p class="new-server__error--info">
        Неверно указан IP адрес. Отредактируйте поле и снова нажмите кнопку Добавить
      </p>
    </div> -->
  </div>
</template>

<script>
import VueIP from './VueIp.vue';
import LoadSpiner from '@/components/forms/LoadSpiner';

export default {
  name: 'NewServer',
  components: {
    VueIP,
    LoadSpiner,
  },
  data: () => ({
    ip_address: '',
    ipValid: null,
    domain_name: '',
    user: '',
    port_ssh: 22,
    port_http: 80,
    port_https: 443,
    loading: false,
  }),
  computed: {
    validForm() {
      return !!(this.ipValid && this.domain_name && this.user && this.port_ssh && this.port_http && this.port_https);
    },
  },
  methods: {
    change(ip, x, valid) {
      this.ip_address = ip;
      this.ipValid = valid;
    },
    async addServer() {
      this.loading = true;
      const { id } = await this.$store.dispatch('servers/addServer', {
        domain_name: this.domain_name,
        ip_address: this.ip_address,
        user: this.user,
        port_ssh: this.port_ssh,
        port_http: this.port_http,
        port_https: this.port_https,
      });
      this.$emit('addServer', id);
      this.loading = false;
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
    > * {
      margin-bottom: 20px;
    }
    &--overlay {
      background: rgba(14, 22, 33, 0.3);
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 10;
    }
  }
  &__ports {
    display: flex;
    gap: 10px;
    > * {
      flex: 1 0 90px;
    }
  }
  &__btn {
    margin-top: 10px;
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